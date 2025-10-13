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

        # Initialize the trainer
        self._setup_logging()
        self._setup_device()
        self._setup_s3()
        self._setup_model()
        self._setup_optimizer()
        self._setup_loss()
        self._setup_data()

    def _setup_logging(self):
        """Setup logging for the trainer"""
        # Create output directory if it doesn't exist
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.args.output_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("S3 Brain-to-Text Trainer initialized")

    def _setup_device(self):
        """Setup the device for training"""
        if torch.cuda.is_available():
            # Get the number of available GPUs
            num_gpus = torch.cuda.device_count()
            self.logger.info(f"Found {num_gpus} GPU(s) available")
            
            # Use the specified GPU number if it exists, otherwise use GPU 0
            requested_gpu = int(self.args.gpu_number)
            if requested_gpu < num_gpus:
                gpu_num = requested_gpu
            else:
                gpu_num = 0
                self.logger.warning(f"Requested GPU {requested_gpu} not available. Using GPU {gpu_num} instead.")
            
            self.device = torch.device(f'cuda:{gpu_num}')
            torch.cuda.set_device(gpu_num)
            self.logger.info(f"Using GPU {gpu_num}: {torch.cuda.get_device_name(gpu_num)}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("CUDA not available. Using CPU")

    def _setup_s3(self):
        """Setup S3 client for checkpoint saving"""
        self.s3_client = boto3.client('s3')
        
        # Set up S3 checkpoint paths
        self.s3_checkpoint_prefix = f"training_results/baseline_rnn/checkpoints/"
        self.s3_best_checkpoint_key = f"{self.s3_checkpoint_prefix}best_checkpoint"
        
        self.logger.info(f"S3 checkpoint prefix: s3://{self.args.dataset.s3_bucket}/{self.s3_checkpoint_prefix}")

    def _setup_model(self):
        """Setup the model"""
        self.model = GRUDecoder(
            neural_dim=self.args.model.n_input_features,
            n_units=self.args.model.n_units,
            n_days=len(self.args.dataset.sessions),
            n_classes=self.args.dataset.n_classes,
            rnn_dropout=self.args.model.rnn_dropout,
            input_dropout=self.args.model.input_network.input_layer_dropout,
            n_layers=self.args.model.n_layers,
            patch_size=self.args.model.patch_size,
            patch_stride=self.args.model.patch_stride
        ).to(self.device)
        
        # Call torch.compile to speed up training (if available)
        try:
            self.model = torch.compile(self.model)
            self.logger.info("Using torch.compile for faster training")
        except Exception as e:
            self.logger.warning(f"torch.compile not available: {e}")
        
        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

    def _setup_optimizer(self):
        """Setup the optimizer"""
        # Main model parameters
        main_params = []
        day_params = []
        
        for name, param in self.model.named_parameters():
            if 'day_specific' in name:
                day_params.append(param)
            else:
                main_params.append(param)
        
        # Create separate optimizers for main model and day-specific layers
        self.optimizer = torch.optim.AdamW([
            {'params': main_params, 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
            {'params': day_params, 'lr': self.args.lr_max_day, 'weight_decay': self.args.weight_decay_day}
        ], betas=(self.args.beta0, self.args.beta1), eps=self.args.epsilon)

        # Setup learning rate scheduler
        if self.args.lr_scheduler_type == 'cosine':
            def lr_lambda(step):
                if step < self.args.lr_warmup_steps:
                    return step / self.args.lr_warmup_steps
                else:
                    progress = (step - self.args.lr_warmup_steps) / (self.args.lr_decay_steps - self.args.lr_warmup_steps)
                    return 0.5 * (1 + math.cos(math.pi * progress))
            
            self.learning_rate_scheduler = LambdaLR(self.optimizer, lr_lambda)

    def _setup_loss(self):
        """Setup the loss function"""
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    def _setup_data(self):
        """Setup the data loaders using S3 direct access"""
        self.logger.info("Setting up S3 data loaders...")
        
        # Create train/test split using S3
        train_trials, test_trials = s3_train_test_split_indicies(
            s3_bucket=self.args.dataset.s3_bucket,
            s3_prefix=self.args.dataset.s3_prefix,
            sessions=self.args.dataset.sessions,
            test_percentage=self.args.dataset.test_percentage,
            seed=self.args.seed,
            bad_trials_dict=self.args.dataset.bad_trials_dict
        )
        
        self.logger.info(f"Created train/test split: {len(train_trials)} training days, {len(test_trials)} test days")
        
        # Create training dataset
        self.train_dataset = S3BrainToTextDataset(
            trial_indicies=train_trials,
            n_batches=self.args.num_training_batches,
            s3_bucket=self.args.dataset.s3_bucket,
            s3_prefix=self.args.dataset.s3_prefix,
            split='train',
            batch_size=self.args.dataset.batch_size,
            days_per_batch=self.args.dataset.days_per_batch,
            random_seed=self.args.seed,
            must_include_days=self.args.dataset.must_include_days,
            feature_subset=self.args.dataset.feature_subset
        )
        
        # Create test dataset
        self.test_dataset = S3BrainToTextDataset(
            trial_indicies=test_trials,
            n_batches=1,  # Will be set automatically for test data
            s3_bucket=self.args.dataset.s3_bucket,
            s3_prefix=self.args.dataset.s3_prefix,
            split='test',
            batch_size=self.args.dataset.batch_size,
            days_per_batch=1,  # Test data uses 1 day per batch
            random_seed=self.args.seed,
            feature_subset=self.args.dataset.feature_subset
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,  # Dataset returns batches, so loader batch_size=1
            shuffle=False,
            num_workers=self.args.dataset.num_dataloader_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.args.dataset.num_dataloader_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.logger.info("Data loaders created successfully")

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.args.num_training_batches} batches")
        self.logger.info(f"Batch size: {self.args.dataset.batch_size}")
        self.logger.info(f"Learning rate: {self.args.lr_max}")
        
        # Initialize training metrics
        train_losses = []
        val_losses = []
        val_pers = []
        
        step = 0
        best_val_per = float('inf')
        
        # Training loop
        for epoch in range(math.ceil(self.args.num_training_batches / len(self.train_loader))):
            self.model.train()
            
            for batch_idx, batch in enumerate(self.train_loader):
                if step >= self.args.num_training_batches:
                    break
                
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Apply data augmentations
                if hasattr(self.args.dataset, 'data_transforms'):
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm_clip_value)
                
                # Update parameters
                self.optimizer.step()
                self.learning_rate_scheduler.step()
                
                # Log training progress
                if step % self.args.batches_per_train_log == 0:
                    current_lr = self.learning_rate_scheduler.get_last_lr()[0]
                    self.logger.info(f"Step {step}/{self.args.num_training_batches}, "
                                   f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
                
                # Validation
                if step % self.args.batches_per_val_step == 0 and step > 0:
                    val_metrics = self.validate()
                    val_losses.append(val_metrics['loss'])
                    val_pers.append(val_metrics['per'])
                    
                    # Save best checkpoint
                    if val_metrics['per'] < best_val_per:
                        best_val_per = val_metrics['per']
                        self.save_checkpoint(step, is_best=True)
                        self.logger.info(f"New best validation PER: {best_val_per:.4f}")
                
                step += 1
            
            if step >= self.args.num_training_batches:
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
            for batch in self.test_loader:
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
        if hasattr(self.args.dataset.data_transforms, 'smooth_data') and self.args.dataset.data_transforms.smooth_data:
            batch['input_features'] = gauss_smooth(
                batch['input_features'],
                kernel_size=self.args.dataset.data_transforms.smooth_kernel_size,
                std=self.args.dataset.data_transforms.smooth_kernel_std
            )
        
        return batch

    def save_checkpoint(self, step, is_best=False):
        """Save model checkpoint to both local and S3"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.learning_rate_scheduler.state_dict(),
            'best_val_per': self.best_val_PER,
            'args': self.args
        }
        
        # Save locally first
        local_checkpoint_path = os.path.join(self.args.checkpoint_dir, f'checkpoint_step_{step}')
        torch.save(checkpoint, local_checkpoint_path)
        
        # Save to S3
        s3_checkpoint_key = f"{self.s3_checkpoint_prefix}checkpoint_step_{step}"
        self._upload_checkpoint_to_s3(checkpoint, s3_checkpoint_key)
        
        # Save best checkpoint
        if is_best:
            # Save locally
            local_best_path = os.path.join(self.args.checkpoint_dir, 'best_checkpoint')
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
                    self.args.dataset.s3_bucket, 
                    s3_key
                )
                
                # Clean up temporary file
                os.unlink(temp_file.name)
                
                self.logger.info(f"Uploaded checkpoint to s3://{self.args.dataset.s3_bucket}/{s3_key}")
                
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
                self.args.dataset.s3_bucket,
                s3_key,
                local_path
            )
            
            # Load checkpoint
            checkpoint = torch.load(local_path, map_location=self.device)
            
            # Clean up temporary file if we created it
            if local_path.startswith('/tmp'):
                os.unlink(local_path)
            
            self.logger.info(f"Loaded checkpoint from s3://{self.args.dataset.s3_bucket}/{s3_key}")
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
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load best validation metrics
        self.best_val_PER = checkpoint.get('best_val_per', torch.inf)
        
        self.logger.info(f"Resumed from step {checkpoint['step']} with best PER: {self.best_val_PER}")
        return checkpoint['step']

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'train_dataset'):
            self.train_dataset.cleanup_cache()
        if hasattr(self, 'test_dataset'):
            self.test_dataset.cleanup_cache()