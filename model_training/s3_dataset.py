import os
import torch
from torch.utils.data import Dataset 
import h5py
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import s3fs
import tempfile
from pathlib import Path

class S3BrainToTextDataset(Dataset):
    '''
    Dataset for brain-to-text data that reads directly from S3
    
    Returns an entire batch of data instead of a single example
    '''

    def __init__(
            self, 
            trial_indicies,
            n_batches,
            s3_bucket,
            s3_prefix,
            split = 'train', 
            batch_size = 64, 
            days_per_batch = 1, 
            random_seed = -1,
            must_include_days = None,
            feature_subset = None,
            cache_dir = None
            ): 
        '''
        trial_indicies:  (dict)      - dictionary with day numbers as keys and lists of trial indices as values
        n_batches:       (int)       - number of random training batches to create
        s3_bucket:       (str)       - S3 bucket name
        s3_prefix:       (str)       - S3 prefix for data files
        split:           (string)    - string specifying if this is a train or test dataset
        batch_size:      (int)       - number of examples to include in batch returned from __getitem_()
        days_per_batch:  (int)       - how many unique days can exist in a batch
        random_seed:     (int)       - seed to set for randomly assigning trials to a batch. If set to -1, trial assignment will be random
        must_include_days ([int])    - list of days that must be included in every batch
        feature_subset  ([int])      - list of neural feature indicies that should be the only features included in the neural data 
        cache_dir       (str)        - local directory to cache downloaded files (optional)
        '''
        
        # Set random seed for reproducibility
        if random_seed != -1:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.split = split
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        
        # Set up S3 filesystem
        self.s3_fs = s3fs.S3FileSystem()
        
        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = tempfile.mkdtemp(prefix='s3_btt_cache_')
        else:
            self.cache_dir = cache_dir
            os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"S3 Dataset initialized:")
        print(f"  Bucket: {s3_bucket}")
        print(f"  Prefix: {s3_prefix}")
        print(f"  Cache directory: {self.cache_dir}")

        # Ensure the split is valid
        if self.split not in ['train', 'test']:
            raise ValueError(f'split must be either "train" or "test". Received {self.split}')
        
        self.days_per_batch = days_per_batch
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.days = {}
        self.n_trials = 0 
        self.trial_indicies = trial_indicies
        self.n_days = len(trial_indicies.keys())
        self.feature_subset = feature_subset

        # Calculate total number of trials in the dataset
        for d in trial_indicies:
            self.n_trials += len(trial_indicies[d]['trials'])

        if must_include_days is not None and len(must_include_days) > days_per_batch:
            raise ValueError(f'must_include_days must be less than or equal to days_per_batch. Received {must_include_days} and days_per_batch {days_per_batch}')
        
        if must_include_days is not None and len(must_include_days) > self.n_days and split != 'train':
            raise ValueError(f'must_include_days is not valid for test data. Received {must_include_days} and but only {self.n_days} in the dataset')
        
        if must_include_days is not None:
            # Map must_include_days to correct indicies if they are negative
            for i, d in enumerate(must_include_days):
                if d < 0: 
                    must_include_days[i] = self.n_days + d

        self.must_include_days = must_include_days    

        # Ensure that the days_per_batch is not greater than the number of days in the dataset. Raise error
        if self.split == 'train' and self.days_per_batch > self.n_days:
            raise ValueError(f'Requested days_per_batch: {days_per_batch} is greater than available days {self.n_days}.')
           
        if self.split == 'train':
            self.batch_index = self.create_batch_index_train()
        else: 
            self.batch_index = self.create_batch_index_test()
            self.n_batches = len(self.batch_index.keys()) # The validation data has a fixed amount of data 
    
    def __len__(self):
        ''' 
        How many batches are in this dataset. 
        Because training data is sampled randomly, there is no fixed dataset length, 
        however this method is required for DataLoader to work 
        '''
        return self.n_batches
    
    def _get_s3_file_path(self, session_path):
        """Convert local session path to S3 path"""
        # Extract session name from path
        session_name = os.path.basename(session_path)
        return f"{self.s3_bucket}/{self.s3_prefix}{session_name}"
    
    def _download_file_if_needed(self, s3_path):
        """Download file from S3 to cache if not already cached"""
        local_path = os.path.join(self.cache_dir, os.path.basename(s3_path))
        
        if not os.path.exists(local_path):
            print(f"Downloading {s3_path} to {local_path}")
            try:
                self.s3_fs.download(s3_path, local_path)
            except Exception as e:
                print(f"Error downloading {s3_path}: {e}")
                raise
        
        return local_path
    
    def __getitem__(self, idx):
        '''
        Returns a batch of data
        '''
        batch = {
            'input_features': [],
            'seq_class_ids': [],
            'transcriptions': [],
            'n_time_steps': [],
            'phone_seq_lens': [],
            'day_indicies': [],
            'block_nums': [],
            'trial_nums': []
        }

        index = self.batch_index[idx]

        # Iterate through each day in the index
        for d in index.keys():
            # Get S3 path for this session
            s3_path = self._get_s3_file_path(self.trial_indicies[d]['session_path'])
            
            # Download file to cache if needed
            local_path = self._download_file_if_needed(s3_path)

            # Open the hdf5 file for that day
            with h5py.File(local_path, 'r') as f:
                # For each trial in the selected trials in that day
                for t in index[d]:
                    try: 
                        g = f[f'trial_{t:04d}']

                        # Remove features if necessary 
                        input_features = torch.from_numpy(g['input_features'][:]) # neural data
                        if self.feature_subset:
                            input_features = input_features[:,self.feature_subset]

                        batch['input_features'].append(input_features)
                        batch['seq_class_ids'].append(torch.from_numpy(g['seq_class_ids'][:]))  # phoneme labels
                        batch['transcriptions'].append(torch.from_numpy(g['transcription'][:])) # character level transcriptions
                        batch['n_time_steps'].append(g.attrs['n_time_steps']) # number of time steps in the trial - required since we are padding
                        batch['phone_seq_lens'].append(g.attrs['seq_len']) # number of phonemes in the label - required since we are padding
                        batch['day_indicies'].append(int(d)) # day index of each trial - required for the day specific layers 
                        batch['block_nums'].append(g.attrs['block_num'])
                        batch['trial_nums'].append(g.attrs['trial_num'])
                    
                    except Exception as e:
                        print(f'Error loading trial {t} from session {s3_path}: {e}')
                        continue

        # Pad data to form a cohesive batch
        batch['input_features'] = pad_sequence(batch['input_features'], batch_first = True, padding_value = 0)
        batch['seq_class_ids'] = pad_sequence(batch['seq_class_ids'], batch_first = True, padding_value = 0)
        batch['transcriptions'] = pad_sequence(batch['transcriptions'], batch_first = True, padding_value = 0)
        batch['n_time_steps'] = torch.tensor(batch['n_time_steps'])
        batch['phone_seq_lens'] = torch.tensor(batch['phone_seq_lens'])
        batch['day_indicies'] = torch.tensor(batch['day_indicies'])
        batch['block_nums'] = torch.tensor(batch['block_nums'])
        batch['trial_nums'] = torch.tensor(batch['trial_nums'])

        return batch

    def create_batch_index_train(self):
        '''
        Creates a dictionary that maps batch indices to the trials that should be included in that batch
        For training data, trials are randomly sampled
        '''
        batch_index = {}
        
        for batch_idx in range(self.n_batches):
            batch_index[batch_idx] = {}
            
            # Randomly select days for this batch
            if self.must_include_days is not None:
                selected_days = self.must_include_days.copy()
                remaining_days = [d for d in self.trial_indicies.keys() if d not in self.must_include_days]
                n_remaining = self.days_per_batch - len(self.must_include_days)
                if n_remaining > 0:
                    additional_days = np.random.choice(remaining_days, size=min(n_remaining, len(remaining_days)), replace=False)
                    selected_days.extend(additional_days)
            else:
                selected_days = np.random.choice(list(self.trial_indicies.keys()), size=self.days_per_batch, replace=False)
            
            # For each selected day, randomly select trials
            for day in selected_days:
                n_trials_in_day = len(self.trial_indicies[day]['trials'])
                n_trials_to_select = min(self.batch_size // self.days_per_batch, n_trials_in_day)
                selected_trials = np.random.choice(self.trial_indicies[day]['trials'], size=n_trials_to_select, replace=False)
                batch_index[batch_idx][day] = selected_trials.tolist()
        
        return batch_index

    def create_batch_index_test(self):
        '''
        Creates a dictionary that maps batch indices to the trials that should be included in that batch
        For test data, trials are deterministically assigned to batches
        '''
        batch_index = {}
        batch_idx = 0
        
        for day in self.trial_indicies.keys():
            trials = self.trial_indicies[day]['trials']
            
            # Split trials into batches
            for i in range(0, len(trials), self.batch_size):
                batch_trials = trials[i:i + self.batch_size]
                batch_index[batch_idx] = {day: batch_trials}
                batch_idx += 1
        
        return batch_index

    def cleanup_cache(self):
        """Clean up cached files"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            print(f"Cleaned up cache directory: {self.cache_dir}")


def s3_train_test_split_indicies(s3_bucket, s3_prefix, sessions, test_percentage=0.1, seed=-1, bad_trials_dict=None):
    """
    Create train/test split indices for S3-based dataset
    """
    import s3fs
    import h5py
    import tempfile
    
    s3_fs = s3fs.S3FileSystem()
    
    # Get all session files from S3
    file_paths = []
    for session in sessions:
        s3_session_path = f"{s3_bucket}/{s3_prefix}{session}"
        try:
            # List files in the session directory
            files = s3_fs.ls(s3_session_path)
            h5_files = [f for f in files if f.endswith('.hdf5')]
            file_paths.extend(h5_files)
        except Exception as e:
            print(f"Warning: Could not access session {session}: {e}")
            continue
    
    print(f"Found {len(file_paths)} HDF5 files in S3")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get trials in each day
        trials_per_day = {}
        for i, s3_path in enumerate(file_paths):
            # Download file temporarily
            local_path = os.path.join(temp_dir, f"temp_{i}.hdf5")
            try:
                s3_fs.download(s3_path, local_path)
                
                # Extract session name from S3 path
                session = [s for s in s3_path.split('/') if (s.startswith('t15.20') or s.startswith('t12.20'))][0]
                
                good_trial_indices = []
                
                with h5py.File(local_path, 'r') as f:
                    num_trials = len(list(f.keys()))
                    for t in range(num_trials):
                        key = f'trial_{t:04d}'
                        
                        block_num = f[key].attrs['block_num']
                        trial_num = f[key].attrs['trial_num']

                        if (
                            bad_trials_dict is not None
                            and session in bad_trials_dict
                            and str(block_num) in bad_trials_dict[session]
                            and trial_num in bad_trials_dict[session][str(block_num)]
                        ):
                            continue

                        good_trial_indices.append(t)

                trials_per_day[i] = {
                    'num_trials': len(good_trial_indices), 
                    'trial_indices': good_trial_indices, 
                    'session_path': s3_path  # Store S3 path instead of local path
                }
                
            except Exception as e:
                print(f"Error processing {s3_path}: {e}")
                continue

        # Pick test_percentage of trials from each day for testing and (1 - test_percentage) for training
        train_trials = {}
        test_trials = {}

        for day in trials_per_day.keys():
            n_trials = trials_per_day[day]['num_trials']
            n_test_trials = int(n_trials * test_percentage)
            
            if seed != -1:
                np.random.seed(seed + day)  # Different seed for each day
            
            # Randomly select test trials
            test_indices = np.random.choice(trials_per_day[day]['trial_indices'], 
                                          size=n_test_trials, 
                                          replace=False)
            train_indices = [t for t in trials_per_day[day]['trial_indices'] 
                           if t not in test_indices]
            
            train_trials[day] = {
                'trials': train_indices,
                'session_path': trials_per_day[day]['session_path']
            }
            test_trials[day] = {
                'trials': test_indices,
                'session_path': trials_per_day[day]['session_path']
            }

        return train_trials, test_trials