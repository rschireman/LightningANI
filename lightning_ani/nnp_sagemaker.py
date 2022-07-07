import sagemaker
from sagemaker.pytorch import PyTorch

# Initializes SageMaker session which holds context data
sagemaker_session = sagemaker.Session()

# The bucket containig our input data
bucket = 's3://dataset.NNP'

role = 'arn:aws:iam::XXXXXX:role/NNP'

# Creates a new PyTorch Estimator 
estimator = PyTorch(
    # name of the runnable script containing __main__ function (entrypoint)
    entry_point='nnp.py',
    # path of the folder containing training code.
    source_dir='./',
    role=role,
    framework_version='1.4.0',
    train_instance_count=1,
    train_instance_type='ml.p2.xlarge',
    # these hyperparameters are passed to the main script as arguments and
    # can be overridden when fine tuning the algorithm
    hyperparameters={
        'max_epochs': 500,
        'batch_size': 1024,
        'learning_rate': 1e-4,
        'data_dir': bucket+'/training'
    })

# Call fit method on estimator, wich trains our model, passing training
# and testing datasets as environment variables. Data is copied from S3
# before initializing the container
estimator.fit({
    'train': bucket+'/training',
})
