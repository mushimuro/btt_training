import os
import torch
import numpy as np
import pandas as pd
import redis
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse
import tempfile
import boto3

from rnn_model import GRUDecoder
from evaluate_model_helpers import *

# --------------------------------------------------------------------------------
# Helper: Download S3 file to local temporary directory
# --------------------------------------------------------------------------------
s3 = boto3.client("s3")

def download_from_s3(s3_path):
    """
    Download a file from S3 (s3://bucket/key) to a temporary local file.
    Returns the local path.
    """
    if not s3_path.startswith("s3://"):
        return s3_path  # already local

    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    tmp_dir = tempfile.mkdtemp()
    local_path = os.path.join(tmp_dir, os.path.basename(key))
    print(f"⬇️ Downloading {s3_path} → {local_path}")
    s3.download_file(bucket, key, local_path)
    return local_path


# --------------------------------------------------------------------------------
# Argument Parser
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Evaluate a pretrained RNN model on the copy task dataset (S3 version).')

parser.add_argument('--model_path', type=str, default='s3://4k-woody-btt/4k/data/t15_pretrained_rnn_baseline',
                    help='S3 path to pretrained model directory.')
parser.add_argument('--data_dir', type=str, default='s3://4k-woody-btt/4k/data/hdf5_data_final',
                    help='S3 path to dataset directory.')
parser.add_argument('--csv_path', type=str, default='s3://4k-woody-btt/4k/data/t15_copyTaskData_description.csv',
                    help='S3 path to the dataset description CSV.')
parser.add_argument('--eval_type', type=str, default='test', choices=['val', 'test'],
                    help='Evaluation type: "val" (with ground truth) or "test".')
parser.add_argument('--gpu_number', type=int, default=1,
                    help='GPU number to use for inference. -1 for CPU.')

args = parser.parse_args()


# --------------------------------------------------------------------------------
# Set up paths
# --------------------------------------------------------------------------------
model_path = args.model_path
data_dir = args.data_dir
eval_type = args.eval_type

# --------------------------------------------------------------------------------
# Load CSV
# --------------------------------------------------------------------------------
csv_local = download_from_s3(args.csv_path)
b2txt_csv_df = pd.read_csv(csv_local)

# --------------------------------------------------------------------------------
# Load model arguments
# --------------------------------------------------------------------------------
args_yaml_s3 = os.path.join(model_path, 'checkpoint/args.yaml')
args_yaml_local = download_from_s3(args_yaml_s3)
model_args = OmegaConf.load(args_yaml_local)

# --------------------------------------------------------------------------------
# Set up GPU device
# --------------------------------------------------------------------------------
gpu_number = args.gpu_number
if torch.cuda.is_available() and gpu_number >= 0:
    if gpu_number >= torch.cuda.device_count():
        raise ValueError(f'GPU number {gpu_number} is out of range. Available GPUs: {torch.cuda.device_count()}')
    device = torch.device(f'cuda:{gpu_number}')
    print(f'Using {device} for model inference.')
else:
    print('Using CPU for model inference.')
    device = torch.device('cpu')

# --------------------------------------------------------------------------------
# Define and load model
# --------------------------------------------------------------------------------
model = GRUDecoder(
    neural_dim=model_args['model']['n_input_features'],
    n_units=model_args['model']['n_units'], 
    n_days=len(model_args['dataset']['sessions']),
    n_classes=model_args['dataset']['n_classes'],
    rnn_dropout=model_args['model']['rnn_dropout'],
    input_dropout=model_args['model']['input_network']['input_layer_dropout'],
    n_layers=model_args['model']['n_layers'],
    patch_size=model_args['model']['patch_size'],
    patch_stride=model_args['model']['patch_stride'],
)

# load model checkpoint from S3
ckpt_s3 = os.path.join(model_path, 'checkpoint/best_checkpoint')
ckpt_local = download_from_s3(ckpt_s3)

checkpoint = torch.load(ckpt_local, map_location=device)
# clean up keys
for key in list(checkpoint['model_state_dict'].keys()):
    new_key = key.replace("module.", "").replace("_orig_mod.", "")
    checkpoint['model_state_dict'][new_key] = checkpoint['model_state_dict'].pop(key)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# --------------------------------------------------------------------------------
# Load dataset from S3
# --------------------------------------------------------------------------------
test_data = {}
total_test_trials = 0

for session in model_args['dataset']['sessions']:
    s3_eval_file = f"{data_dir}/{session}/data_{eval_type}.hdf5"
    eval_file_local = download_from_s3(s3_eval_file)

    data = load_h5py_file(eval_file_local, b2txt_csv_df)
    test_data[session] = data
    total_test_trials += len(data["neural_features"])
    print(f'✅ Loaded {len(data["neural_features"])} {eval_type} trials for session {session}.')

print(f'Total number of {eval_type} trials: {total_test_trials}\n')

# --------------------------------------------------------------------------------
# Predict phoneme sequences
# --------------------------------------------------------------------------------
with tqdm(total=total_test_trials, desc='Predicting phoneme sequences', unit='trial') as pbar:
    for session, data in test_data.items():
        data['logits'] = []
        input_layer = model_args['dataset']['sessions'].index(session)

        for trial in range(len(data['neural_features'])):
            neural_input = data['neural_features'][trial]
            neural_input = np.expand_dims(neural_input, axis=0)
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)
            logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
            data['logits'].append(logits)
            pbar.update(1)
pbar.close()

# --------------------------------------------------------------------------------
# Convert logits to phoneme sequences
# --------------------------------------------------------------------------------
for session, data in test_data.items():
    data['pred_seq'] = []
    for trial in range(len(data['logits'])):
        logits = data['logits'][trial][0]
        pred_seq = np.argmax(logits, axis=-1)
        pred_seq = [int(p) for p in pred_seq if p != 0]
        pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i-1]]
        pred_seq = [LOGIT_TO_PHONEME[p] for p in pred_seq]
        data['pred_seq'].append(pred_seq)

        block_num = data['block_num'][trial]
        trial_num = data['trial_num'][trial]
        print(f'Session: {session}, Block: {block_num}, Trial: {trial_num}')
        if eval_type == 'val':
            sentence_label = data['sentence_label'][trial]
            true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
            true_seq = [LOGIT_TO_PHONEME[p] for p in true_seq]
            print(f'Sentence label:      {sentence_label}')
            print(f'True sequence:       {" ".join(true_seq)}')
        print(f'Predicted Sequence:  {" ".join(pred_seq)}\n')

# --------------------------------------------------------------------------------
# Redis Language Model Inference
# --------------------------------------------------------------------------------
r = redis.Redis(host='localhost', port=6379, db=0)
r.flushall()

remote_lm_input_stream = 'remote_lm_input'
remote_lm_output_partial_stream = 'remote_lm_output_partial'
remote_lm_output_final_stream = 'remote_lm_output_final'

remote_lm_output_partial_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_output_final_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_resetting_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_finalizing_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_updating_lastEntrySeen = get_current_redis_time_ms(r)

lm_results = {'session': [], 'block': [], 'trial': [], 'true_sentence': [], 'pred_sentence': []}

with tqdm(total=total_test_trials, desc='Running remote language model', unit='trial') as pbar:
    for session in test_data.keys():
        for trial in range(len(test_data[session]['logits'])):
            logits = rearrange_speech_logits_pt(test_data[session]['logits'][trial])[0]
            remote_lm_done_resetting_lastEntrySeen = reset_remote_language_model(r, remote_lm_done_resetting_lastEntrySeen)
            remote_lm_output_partial_lastEntrySeen, _ = send_logits_to_remote_lm(
                r, remote_lm_input_stream, remote_lm_output_partial_stream,
                remote_lm_output_partial_lastEntrySeen, logits
            )
            remote_lm_output_final_lastEntrySeen, lm_out = finalize_remote_lm(
                r, remote_lm_output_final_stream, remote_lm_output_final_lastEntrySeen
            )

            best_candidate_sentence = lm_out['candidate_sentences'][0]
            lm_results['session'].append(session)
            lm_results['block'].append(test_data[session]['block_num'][trial])
            lm_results['trial'].append(test_data[session]['trial_num'][trial])
            lm_results['true_sentence'].append(
                test_data[session]['sentence_label'][trial] if eval_type == 'val' else None
            )
            lm_results['pred_sentence'].append(best_candidate_sentence)
            pbar.update(1)
pbar.close()

# --------------------------------------------------------------------------------
# WER Evaluation (if validation)
# --------------------------------------------------------------------------------
if eval_type == 'val':
    total_true_length = 0
    total_edit_distance = 0
    lm_results['edit_distance'] = []
    lm_results['num_words'] = []

    for i in range(len(lm_results['pred_sentence'])):
        true_sentence = remove_punctuation(lm_results['true_sentence'][i]).strip()
        pred_sentence = remove_punctuation(lm_results['pred_sentence'][i]).strip()
        ed = editdistance.eval(true_sentence.split(), pred_sentence.split())
        total_true_length += len(true_sentence.split())
        total_edit_distance += ed
        lm_results['edit_distance'].append(ed)
        lm_results['num_words'].append(len(true_sentence.split()))
        print(f'{lm_results["session"][i]} - Block {lm_results["block"][i]}, Trial {lm_results["trial"][i]}')
        print(f'True sentence:       {true_sentence}')
        print(f'Predicted sentence:  {pred_sentence}')
        print(f'WER: {100 * ed / len(true_sentence.split()):.2f}%\n')

    print(f'Aggregate WER: {100 * total_edit_distance / total_true_length:.2f}%')

# --------------------------------------------------------------------------------
# Save results
# --------------------------------------------------------------------------------
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(tempfile.gettempdir(), f'baseline_rnn_{eval_type}_predicted_sentences_{timestamp}.csv')
ids = list(range(len(lm_results['pred_sentence'])))
df_out = pd.DataFrame({'id': ids, 'text': lm_results['pred_sentence']})
df_out.to_csv(output_file, index=False)
print(f"💾 Results saved to {output_file}")
