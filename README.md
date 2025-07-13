# Self documentation for my custom machine learning model

# CURRENTLY DOES NOT WORK

## Trained on the DeepFashion2 dataset

## Make sure you have gcloud, conda, and pip

### Assuming you have instantiated a cluster on Ray on Vertex AI with Ray version 2.42 and Python version 3.10
### use ray --version and python --version in your custom environment variable to double check

$ gcloud auth login

$ conda create -n py310 python=3.10.18 -y

$ conda activate py310

$ pip install -r requirements.txt

Debugging

1. conda environment showing wrong python version

Solution:
To remove a conda environment, view it and remove it via
$ conda env list
$ conda remove --name CONDA_ENV --all
$ env:PATH = "C:\Users\JJjer\miniconda3\envs\py310;C:\Users\JJjer\miniconda3\envs\py310\Scripts;C:\Users\JJjer\miniconda3\envs\py310\Library\bin;" + $env:PATH

2. Failure in attempting to start up cluster
[2025-07-12 23:15:45,813 W ...] gcs_rpc_client.h:148: Failed to connect to GCS at address 10.126.20.4:2222 within 5 seconds.
[2025-07-12 23:15:50,857 W ...] gcs_client.cc:178: Failed to get cluster ID from GCS server: TimedOut

Solution:
ray up -y .\ray-cluster-config.yaml
ray down -y .\ray-cluster-config.yaml
ray up -y .\ray-cluster-config.yaml

2.1. Timeout, server 34.94.194.79 not responding.
    SSH still not available (SSH command failed.), retrying in 5 seconds.

Solution:
$ taskkill /f /im python.exe
$ gcloud compute firewall-rules create allow-ssh-ray --network=default  --allow tcp:22  --source-ranges=0.0.0.0/0  --description="Allow SSH connections for Ray cluster"
$ gcloud compute firewall-rules update allow-ssh-ray --target-tags=deeplearning-vm
