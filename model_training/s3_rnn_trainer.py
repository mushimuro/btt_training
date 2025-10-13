import torch 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import random
import time
import os
import numpy as np
import math
import pathlib
import logging
import sys
import json
import pickle
import boto3
import tempfile
from botocore.exceptions import ClientError

# Import the S3 dataset instead of regular dataset
from s3_dataset import S3BrainToTextDataset, s3_train_test_split_indicies
from data_augmentations import gauss_smooth

import torchaudio.functional as F # for edit distance
from omegaconf import OmegaConf

torch.set_float32_matmul_precision('high') # makes float32 matmuls faster on some GPUs
torch.backends.cudnn.deterministic = True # makes training more reproducible
torch._dynamo.config.cache_size_limit = 64

from rnn_model import GRUDecoder

class S3BrainToTextDecoder_Trainer:
    """
    This class will initialize and train a brain-to-text phoneme decoder using S3 direct access
    
    Modified from the original BrainToTextDecoder_Trainer to work with S3
    Written by Nick Card and Zachery Fogg with reference to Stanford NPTL's decoding function
    """

    def __init__(self, args):
        '''
        args : dictionary of training arguments
        '''

        # Trainer fields
        self.args = args
        self.logger = None 
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.ctc_loss = None 

        self.best_val_PER = torch.inf # track best PER for checkpointing
        self.best_val_loss = torch.inf # track best loss for checkpointing

        self.train_dataset = None 
        self.val_dataset = None 
        self.train_loader = None 
        self.val_loader = None 

        self.transform_args = self.args['dataset']['data_transforms']

        # Create output directory
        if args['mode'] == 'train':
            os.makedirs(self.args['output_dir'], exist_ok=False)

        # Create checkpoint directory
        if args['save_best_checkpoint'] or args['save_all_val_steps'] or args['save_final_model']: 
            os.makedirs(self.args['checkpoint_dir'], exist_ok=False)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]:  # make a copy of the list
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')        

        if args['mode']=='train':
            # During training, save logs to file in output directory
            fh = logging.FileHandler(str(pathlib.Path(self.args['output_dir'],'training_log')))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # Always print logs to stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # Configure device pytorch will use 
        if torch.cuda.is_available():
            gpu_num = self.args.get('gpu_number', 0)
            try:
                gpu_num = int(gpu_num)
            except ValueError:
                self.logger.warning(f"Invalid gpu_number value: {gpu_num}. Using 0 instead.")
                gpu_num = 0

            max_gpu_index = torch.cuda.device_count() - 1
            if gpu_num > max_gpu_index:
                self.logger.warning(f"Requested GPU {gpu_num} not available. Using GPU 0 instead.")
                gpu_num = 0

            try:
                self.device = torch.device(f"cuda:{gpu_num}")
                test_tensor = torch.tensor([1.0]).to(self.device)
                test_tensor = test_tensor * 2
            except Exception as e:
                self.logger.error(f"Error initializing CUDA device {gpu_num}: {str(e)}")
                self.logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f'Using device: {self.device}')

        # Set seed if provided 
        if self.args['seed'] != -1:
            np.random.seed(self.args['seed'])
            random.seed(self.args['seed'])
            torch.manual_seed(self.args['seed'])

        # Initialize the model 
        self.model = GRUDecoder(
            neural_dim = self.args['model']['n_input_features'],
            n_units = self.args['model']['n_units'],
            n_days = len(self.args['dataset']['sessions']),
            n_classes  = self.args['dataset']['n_classes'],
            rnn_dropout = self.args['model']['rnn_dropout'], 
            input_dropout = self.args['model']['input_network']['input_layer_dropout'], 
            n_layers = self.args['model']['n_layers'],
            patch_size = self.args['model']['patch_size'],
            patch_stride = self.args['model']['patch_stride'],
        )

        # Call torch.compile to speed up training
        self.logger.info("Using torch.compile")
        self.model = torch.compile(self.model)

        self.logger.info(f"Initialized RNN decoding model")

        self.logger.info(self.model)

        # Log how many parameters are in the model
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")

        # Determine how many day-specific parameters are in the model
        day_params = 0
        for name, param in self.model.named_parameters():
            if 'day' in name:
                day_params += param.numel()
        
        self.logger.info(f"Model has {day_params:,} day-specific parameters | {((day_params / total_params) * 100):.2f}% of total parameters")

        # Setup S3 client for checkpoint saving
        self.s3_client = boto3.client('s3')
        self.s3_checkpoint_prefix = f"training_results/baseline_rnn/checkpoints/"
        self.s3_best_checkpoint_key = f"{self.s3_checkpoint_prefix}best_checkpoint"
        self.logger.info(f"S3 checkpoint prefix: s3://{self.args['dataset']['s3_bucket']}/{self.s3_checkpoint_prefix}")

        # Create datasets and dataloaders using S3
        self._setup_s3_datasets()

        # Create optimizer, learning rate scheduler, and loss
        self.optimizer = self.create_optimizer()

        if self.args['lr_scheduler_type'] == 'linear':
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer = self.optimizer,
                start_factor = 1.0,
                end_factor = self.args['lr_min'] / self.args['lr_max'],
                total_iters = self.args['lr_decay_steps'],
            )
        elif self.args['lr_scheduler_type'] == 'cosine':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)
        
        else:
            raise ValueError(f"Invalid learning rate scheduler type: {self.args['lr_scheduler_type']}")
        
        self.ctc_loss = torch.nn.CTCLoss(blank = 0, reduction = 'none', zero_infinity = False)

        # If a checkpoint is provided, then load from checkpoint
        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

        # Set rnn and/or input layers to not trainable if specified 
        for name, param in self.model.named_parameters():
            if not self.args['model']['rnn_trainable'] and 'gru' in name:
                param.requires_grad = False

            elif not self.args['model']['input_network']['input_trainable'] and 'day' in name:
                param.requires_grad = False

        # Send model to device 
        self.model.to(self.device)

    def _setup_s3_datasets(self):
        """Setup datasets using S3 direct access"""
        self.logger.info("Setting up S3 datasets...")
        
        # Split trials into train and test sets using S3
        train_trials, val_trials = s3_train_test_split_indicies(
            s3_bucket=self.args['dataset']['s3_bucket'],
            s3_prefix=self.args['dataset']['s3_prefix'],
            sessions=self.args['dataset']['sessions'],
            test_percentage=self.args['dataset']['test_percentage'],
            seed=self.args['dataset']['seed'],
            bad_trials_dict=self.args['dataset']['bad_trials_dict']
        )

        # Save dictionaries to output directory to know which trials were train vs val 
        with open(os.path.join(self.args['output_dir'], 'train_val_trials.json'), 'w') as f: 
            json.dump({'train' : train_trials, 'val': val_trials}, f)

        # Determine if a only a subset of neural features should be used
        feature_subset = None
        if ('feature_subset' in self.args['dataset']) and self.args['dataset']['feature_subset'] != None: 
            feature_subset = self.args['dataset']['feature_subset']
            self.logger.info(f'Using only a subset of features: {feature_subset}')
            
        # train dataset and dataloader
        self.train_dataset = S3BrainToTextDataset(
            trial_indicies = train_trials,
            n_batches = self.args['num_training_batches'],
            s3_bucket = self.args['dataset']['s3_bucket'],
            s3_prefix = self.args['dataset']['s3_prefix'],
            split = 'train',
            batch_size = self.args['dataset']['batch_size'],
            days_per_batch = self.args['dataset']['days_per_batch'],
            random_seed = self.args['dataset']['seed'],
            must_include_days = None,
            feature_subset = feature_subset
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size = None, # Dataset.__getitem__() already returns batches
            shuffle = self.args['dataset']['loader_shuffle'],
            num_workers = self.args['dataset']['num_dataloader_workers'],
            pin_memory = True 
        )

        # val dataset and dataloader
        self.val_dataset = S3BrainToTextDataset(
            trial_indicies = val_trials,
            n_batches = None,
            s3_bucket = self.args['dataset']['s3_bucket'],
            s3_prefix = self.args['dataset']['s3_prefix'],
            split = 'test',
            batch_size = self.args['dataset']['batch_size'],
            days_per_batch = None,
            random_seed = self.args['dataset']['seed'],
            must_include_days = None,
            feature_subset = feature_subset   
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size = None, # Dataset.__getitem__() already returns batches
            shuffle = False, 
            num_workers = 0,
            pin_memory = True 
        )

        self.logger.info("Successfully initialized S3 datasets")

    def create_optimizer(self):
        '''
        Create the optimizer with special param groups 

        Biases and day weights should not be decayed

        Day weights should have a separate learning rate
        '''
        bias_params = [p for name, p in self.model.named_parameters() if 'gru.bias' in name or 'out.bias' in name]
        day_params = [p for name, p in self.model.named_parameters() if 'day_' in name]
        other_params = [p for name, p in self.model.named_parameters() if 'day_' not in name and 'gru.bias' not in name and 'out.bias' not in name]

        if len(day_params) != 0:
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : day_params, 'lr' : self.args['lr_max_day'], 'weight_decay' : self.args['weight_decay_day'], 'group_type' : 'day_layer'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
        else: 
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
            
        optim = torch.optim.AdamW(
            param_groups,
            lr = self.args['lr_max'],
            betas = (self.args['beta0'], self.args['beta1']),
            eps = self.args['epsilon'],
            weight_decay = self.args['weight_decay'],
            fused = True
        )

        return optim 

    def create_cosine_lr_scheduler(self, optim):
        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_decay_steps = self.args['lr_decay_steps']

        lr_max_day =  self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']
        lr_decay_steps_day = self.args['lr_decay_steps_day']

        lr_warmup_steps = self.args['lr_warmup_steps']
        lr_warmup_steps_day = self.args['lr_warmup_steps_day']

        def lr_lambda(step):
            if step < lr_warmup_steps:
                return step / lr_warmup_steps
            else:
                progress = (step - lr_warmup_steps) / (lr_decay_steps - lr_warmup_steps)
                return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))

        def lr_lambda_day(step):
            if step < lr_warmup_steps_day:
                return step / lr_warmup_steps_day
            else:
                progress = (step - lr_warmup_steps_day) / (lr_decay_steps_day - lr_warmup_steps_day)
                return lr_min_day + (lr_max_day - lr_min_day) * 0.5 * (1 + math.cos(math.pi * progress))

        # Create separate schedulers for different parameter groups
        schedulers = []
        for i, group in enumerate(optim.param_groups):
            if group['group_type'] == 'day_layer':
                scheduler = LambdaLR(optim, lr_lambda_day)
            else:
                scheduler = LambdaLR(optim, lr_lambda)
            schedulers.append(scheduler)

        return schedulers

    def train(self):
        """
        Main training loop
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.args['num_training_batches']} batches")
        self.logger.info(f"Batch size: {self.args['dataset']['batch_size']}")
        self.logger.info(f"Learning rate: {self.args['lr_max']}")
        
        # Initialize training metrics
        train_losses = []
        val_losses = []
        val_pers = []
        
        step = 0
        best_val_per = float('inf')
        
        # Training loop
        for epoch in range(math.ceil(self.args['num_training_batches'] / len(self.train_loader))):
            self.model.train()
            
            for batch_idx, batch in enumerate(self.train_loader):
                if step >= self.args['num_training_batches']:
                    break
                
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Apply data augmentations
                if hasattr(self.args['dataset'], 'data_transforms'):
                    batch = self._apply_data_transforms(batch)
                
                # Get model outputs
                logits = self.model(
                    batch['input_features'],
                    batch['day_indicies']
                )
                
                # Compute loss
                loss = self.ctc_loss(
                    logits.transpose(0, 1),  # (seq_len, batch_size, n_classes)
                    batch['seq_class_ids'],
                    batch['n_time_steps'],
                    batch['phone_seq_lens']
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['grad_norm_clip_value'])
                
                # Update parameters
                self.optimizer.step()
                
                # Update learning rate schedulers
                if isinstance(self.learning_rate_scheduler, list):
                    for scheduler in self.learning_rate_scheduler:
                        scheduler.step()
                else:
                    self.learning_rate_scheduler.step()
                
                # Log training progress
                if step % self.args['batches_per_train_log'] == 0:
                    if isinstance(self.learning_rate_scheduler, list):
                        current_lr = self.learning_rate_scheduler[0].get_last_lr()[0]
                    else:
                        current_lr = self.learning_rate_scheduler.get_last_lr()[0]
                    self.logger.info(f"Step {step}/{self.args['num_training_batches']}, "
                                   f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
                
                # Validation
                if step % self.args['batches_per_val_step'] == 0 and step > 0:
                    val_metrics = self.validate()
                    val_losses.append(val_metrics['loss'])
                    val_pers.append(val_metrics['per'])
                    
                    # Save best checkpoint
                    if val_metrics['per'] < best_val_per:
                        best_val_per = val_metrics['per']
                        self.save_checkpoint(step, is_best=True)
                        self.logger.info(f"New best validation PER: {best_val_per:.4f}")
                
                step += 1
            
            if step >= self.args['num_training_batches']:
                break
        
        # Final validation
        final_metrics = self.validate()
        self.logger.info(f"Training completed. Final validation PER: {final_metrics['per']:.4f}")
        
        return {
            'final_val_loss': final_metrics['loss'],
            'final_val_per': final_metrics['per'],
            'best_val_per': best_val_per
        }

    def validate(self):
        """Run validation"""
        self.model.eval()
        total_loss = 0
        total_per = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                logits = self.model(
                    batch['input_features'],
                    batch['day_indicies']
                )
                
                # Compute loss
                loss = self.ctc_loss(
                    logits.transpose(0, 1),
                    batch['seq_class_ids'],
                    batch['n_time_steps'],
                    batch['phone_seq_lens']
                )
                
                # Compute PER (simplified - you might want to implement proper PER calculation)
                predictions = torch.argmax(logits, dim=-1)
                per = self._compute_per(predictions, batch['seq_class_ids'])
                
                total_loss += loss.item()
                total_per += per
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_per = total_per / num_batches
        
        self.logger.info(f"Validation - Loss: {avg_loss:.4f}, PER: {avg_per:.4f}")
        
        return {'loss': avg_loss, 'per': avg_per}

    def _compute_per(self, predictions, targets):
        """Compute Phoneme Error Rate (simplified version)"""
        # This is a simplified PER calculation
        # You might want to implement a more sophisticated version
        return 0.5  # Placeholder

    def _apply_data_transforms(self, batch):
        """Apply data augmentations"""
        if hasattr(self.args['dataset']['data_transforms'], 'smooth_data') and self.args['dataset']['data_transforms']['smooth_data']:
            batch['input_features'] = gauss_smooth(
                batch['input_features'],
                kernel_size=self.args['dataset']['data_transforms']['smooth_kernel_size'],
                std=self.args['dataset']['data_transforms']['smooth_kernel_std']
            )
        
        return batch

    def save_checkpoint(self, step, is_best=False):
        """Save model checkpoint to both local and S3"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': [s.state_dict() for s in self.learning_rate_scheduler] if isinstance(self.learning_rate_scheduler, list) else self.learning_rate_scheduler.state_dict(),
            'best_val_per': self.best_val_PER,
            'args': self.args
        }
        
        # Save locally first
        local_checkpoint_path = os.path.join(self.args['checkpoint_dir'], f'checkpoint_step_{step}')
        torch.save(checkpoint, local_checkpoint_path)
        
        # Save to S3
        s3_checkpoint_key = f"{self.s3_checkpoint_prefix}checkpoint_step_{step}"
        self._upload_checkpoint_to_s3(checkpoint, s3_checkpoint_key)
        
        # Save best checkpoint
        if is_best:
            # Save locally
            local_best_path = os.path.join(self.args['checkpoint_dir'], 'best_checkpoint')
            torch.save(checkpoint, local_best_path)
            
            # Save to S3
            self._upload_checkpoint_to_s3(checkpoint, self.s3_best_checkpoint_key)
            self.logger.info(f"Saved best checkpoint at step {step} to both local and S3")

    def _upload_checkpoint_to_s3(self, checkpoint, s3_key):
        """Upload checkpoint to S3"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
                torch.save(checkpoint, temp_file.name)
                
                # Upload to S3
                self.s3_client.upload_file(
                    temp_file.name, 
                    self.args['dataset']['s3_bucket'], 
                    s3_key
                )
                
                # Clean up temporary file
                os.unlink(temp_file.name)
                
                self.logger.info(f"Uploaded checkpoint to s3://{self.args['dataset']['s3_bucket']}/{s3_key}")
                
        except ClientError as e:
            self.logger.error(f"Failed to upload checkpoint to S3: {e}")
            raise

    def load_checkpoint_from_s3(self, s3_key, local_path=None):
        """Load checkpoint from S3"""
        try:
            if local_path is None:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
                    local_path = temp_file.name
            
            # Download from S3
            self.s3_client.download_file(
                self.args['dataset']['s3_bucket'],
                s3_key,
                local_path
            )
            
            # Load checkpoint
            checkpoint = torch.load(local_path, map_location=self.device)
            
            # Clean up temporary file if we created it
            if local_path.startswith('/tmp'):
                os.unlink(local_path)
            
            self.logger.info(f"Loaded checkpoint from s3://{self.args['dataset']['s3_bucket']}/{s3_key}")
            return checkpoint
            
        except ClientError as e:
            self.logger.error(f"Failed to load checkpoint from S3: {e}")
            raise

    def resume_from_s3_checkpoint(self, s3_checkpoint_key):
        """Resume training from an S3 checkpoint"""
        self.logger.info(f"Resuming training from S3 checkpoint: {s3_checkpoint_key}")
        
        # Load checkpoint from S3
        checkpoint = self.load_checkpoint_from_s3(s3_checkpoint_key)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if isinstance(self.learning_rate_scheduler, list):
            for i, scheduler in enumerate(self.learning_rate_scheduler):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'][i])
        else:
            self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load best validation metrics
        self.best_val_PER = checkpoint.get('best_val_per', torch.inf)
        
        self.logger.info(f"Resumed from step {checkpoint['step']} with best PER: {self.best_val_PER}")
        return checkpoint['step']

    def load_model_checkpoint(self, checkpoint_path):
        """Load model from checkpoint (same as original)"""
        self.logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            if isinstance(self.learning_rate_scheduler, list):
                for i, scheduler in enumerate(self.learning_rate_scheduler):
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'][i])
            else:
                self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load best validation metrics if available
        if 'best_val_per' in checkpoint:
            self.best_val_PER = checkpoint['best_val_per']
        
        self.logger.info(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'train_dataset'):
            self.train_dataset.cleanup_cache()
        if hasattr(self, 'val_dataset'):
            self.val_dataset.cleanup_cache()