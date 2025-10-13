# AWS SageMaker Setup for Brain-to-Text Training

This guide explains how to use the `sagemaker_brain_to_text_training.ipynb` notebook to train the RNN baseline model on AWS SageMaker with **direct S3 access**.

## Key Benefits of S3 Direct Access

- **No Data Download**: Training data is accessed directly from S3, no need to download entire dataset
- **Faster Startup**: No waiting time for data download
- **Less Storage**: Minimal local storage requirements
- **Persistent**: No data loss when notebook instance stops
- **Cost Effective**: Reduced storage costs and faster training cycles

## Prerequisites

1. **AWS Account**: You need an AWS account with appropriate permissions
2. **SageMaker Notebook Instance**: Create a SageMaker notebook instance with:
   - Instance type: `ml.g4dn.xlarge` or larger (for GPU training)
   - Python 3 kernel
   - At least 20GB of storage (reduced from 50GB due to S3 direct access)
3. **S3 Bucket Access**: Ensure your SageMaker instance has access to the `4k-woody-btt` S3 bucket

## IAM Permissions

Your SageMaker execution role needs the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::4k-woody-btt",
                "arn:aws:s3:::4k-woody-btt/*"
            ]
        }
    ]
}
```

## Setup Instructions

### 1. Upload the Notebook

1. Open your SageMaker notebook instance
2. Upload `sagemaker_brain_to_text_training.ipynb` to your notebook instance
3. Open the notebook in Jupyter

### 2. Configure S3 Access

The notebook is pre-configured to use:
- **S3 Bucket**: `4k-woody-btt`
- **Data Path**: `4k/data/`
- **Access Method**: Direct S3 access (no local download)

### 3. Run the Notebook

Execute the cells in order:

1. **Install Dependencies**: Installs required packages including AWS SDK and S3 filesystem
2. **Import Libraries**: Sets up PyTorch and AWS clients
3. **S3 Configuration**: Configures S3 bucket and paths
4. **S3 Direct Access Setup**: Scans S3 for available training data files
5. **Clone Repository**: Clones the brain-to-text repository
6. **Configure Training**: Sets up training parameters for S3 direct access
7. **Start Training**: Runs the RNN baseline training with S3 direct access
8. **Upload Results**: Optionally uploads trained models back to S3

## Key Differences from Kaggle Version

| Aspect | Kaggle Version | SageMaker Version |
|--------|----------------|-------------------|
| Data Source | `/kaggle/input/` | S3 bucket `4k-woody-btt/4k/data/` |
| Data Access | Local files | Direct S3 access (streaming) |
| Storage | Kaggle workspace | Minimal local cache + SageMaker storage |
| Output | Kaggle output | Local + S3 upload |
| Paths | `/kaggle/working/` | `/home/ec2-user/SageMaker/` |
| Startup Time | Fast | Very fast (no download) |

## Configuration Options

### Training Parameters

The notebook uses the same configuration as the original `rnn_args.yaml` but with S3 direct access:

- **S3 Bucket**: `4k-woody-btt`
- **S3 Prefix**: `4k/data/`
- **Use S3 Direct**: `True` (enables direct S3 access)
- **Output Directory**: `/home/ec2-user/SageMaker/trained_models/baseline_rnn`
- **Checkpoint Directory**: `/home/ec2-user/SageMaker/trained_models/baseline_rnn/checkpoint`

### Resume Training

To resume from a checkpoint, uncomment and modify these lines in the configuration cell:

```python
args.init_from_checkpoint = True
args.init_checkpoint_path = '/home/ec2-user/SageMaker/trained_models/baseline_rnn/checkpoint/best_checkpoint'
```

## Monitoring Training

The training process will output:
- Training loss and metrics
- Validation performance
- Checkpoint saves
- Final training summary

## Output Files

After training, the following files will be created:

- **Model Checkpoints**: Saved in the checkpoint directory
- **Training Logs**: Detailed training metrics
- **Final Model**: Best performing model
- **S3 Upload**: Results uploaded to `s3://4k-woody-btt/training_results/baseline_rnn/`

## Troubleshooting

### Common Issues

1. **S3 Access Denied**: Check IAM permissions for the SageMaker execution role
2. **Out of Memory**: Use a larger instance type (e.g., `ml.g4dn.2xlarge`)
3. **S3 Connection Issues**: Verify S3 bucket access and network connectivity
4. **CUDA Issues**: Verify GPU availability with `torch.cuda.is_available()`
5. **Cache Directory Issues**: Ensure write permissions for temporary cache directory

### Performance Tips

1. **Use GPU Instances**: Training is much faster with GPU acceleration
2. **Increase Batch Size**: If you have more memory, increase `batch_size` in config
3. **Parallel Data Loading**: Adjust `num_dataloader_workers` based on instance specs

## Cost Optimization

- **Spot Instances**: Use SageMaker spot instances for cost savings
- **Auto-Stop**: Configure notebook instance to auto-stop when idle
- **S3 Direct Access**: No data download means faster startup and lower storage costs
- **Cache Management**: Automatic cleanup of temporary cached files

## Next Steps

After successful training:

1. Download trained models from S3
2. Use the models for inference
3. Evaluate performance on test data
4. Consider hyperparameter tuning for better results