# MKFL: practical tool for Secure Federated Learning

MKFL is a privacy-preserving federated learning tool based on the xMK-CKKS encryption scheme. This repository is a supplementary materials for the paper "Ensuring Data Privacy with MKFL, a practical tool for Secure Federated Learning".

- *Client data security* is provided by the [xMK-CKKS homomorphic multi-key encryption scheme](https://arxiv.org/abs/2104.06824) \[1\]. The parameters communicated between the clients and the server are encrypted using the aggregated public key and can only be decrypted by the collaboration of all clients.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Docker](#docker)
- [Results](#results)
- [Examples](#examples)


## Installation

1. Install the dependencies *flower* and *TenSEAL*:


```bash
cd tools/flower
pip install .
```

```bash
cd tools/TenSEAL
pip install .
```

2. Install dependencies:
```bash
 pip install -r requirements.txt
 pip install pyg_lib torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
 pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```
 
## Usage
### Scripts
The **MKFL/src/FL** folder contains bash scripts to run easily the federated learning task on a single machine:
 - *fl_run.sh* : without encryption and without contribution evaluation
 - *fl_run_enc.sh* : with encryption but without contribution evaluation

Do not forget to replace the address of the server at the end of the *src/FL/fl_client_enc.py*.
 
### Manually
If you want to run each component manually, you need to run the aggregation server and the clients.

To run the secure server, use the following command in **MKFL/src/**:
```python
python FL/fl_server_enc.py --nclients [number of clients] --nrounds [number of rounds] --filepath [folder path to store results] --dataset [split_scdg1] --model [GINE]
```
To run the clients, use the following command in **MKFL/src/**:
```python
python FL/fl_client_enc.py --nclients [number of clients] --partition [client_number] --filepath [folder path to store results] --dataset [split_scdg1] --model [GINE]
```

To run the federated learning task without encryption, replace the filenames of the server and the clients by *fl_server.py* and *fl_client.py*, respectively.

Do not forget to replace the address of the server at the end of the *src/FL/fl_client_enc.py*.

### Modify the database used

In the *init_datasets* function of *src/FL/main_utils*:
- add a branch with the name of the dataset
- create a Dataset object and provide it an initialization function
- call init_db on the new Dataset and return it


### Modify the model used

In the *get_model* function of *src/FL/main_utils*:
- add a branch with the name of the model
- initialize your model
- create a Model object and provide your model to it
- return the Model object


## Docker Installation and Usage

Docker image can be created with
```bash
docker build ./ -t mkfl
```

On a single machine, **secure** federated learning framework can be run with docker compose:
```bash
docker compose up -d
```
*.env* file contains parameters for docker compose: number of clients, number of rounds, dataset name, model name, path to the dataset, and path to a folder to output results.

To run on multiple machines, first, the server has to be started:
```bash
docker run -d -v [folder path to store results]:/results -v [folder path with databases]:/mkfl/databases mkfl /bin/bash FL/docker_se.sh [number of clients] [number of rounds] [dataset] [model] [is_encrypted] [server_IP]
```
where [is_encrypted] should be *true* for secure aggregation and *false* for the federated learning without encryption.

Next, each client is launched with the same image (docker image can be distributed with *docker save* and *docker load* or by setting up a local docker image repository).
```bash
docker run -d -v [folder path to store results]:/results -v [folder path with databases]:/mkfl/databases mkfl /bin/bash FL/docker_cl.sh [number of clients] [number of rounds] [dataset] [model] [is_encrypted] [client_number] [server_IP]
```

Note that for both docker and docker compose, there is no need to replace the address of the server in the file, it is done automatically.

To check logs:
```bash
docker container logs [container_id]
```

## Results

Results are stored in the folder specified by the *filepath* argument.
In the specified folder, a new folder, having as name the time of the beginning of the federated learning process, is created. In this folder are saved the parameters generated at each round and a *csv* file containing some metrics about the experiments.

For the client, the received aggregated parameters are saved named as *global* while the newly generated parameters are saved named as *local*. The server only saves the aggregated parameters. 
For the client, the *csv* file contains the following metrics: model, accuracy, precision, recall, f1 score, balanced accuracy, loss, train time, test time and the predicted labels.
These metrics are evaluated by the model using the received parameters and then by the new model after fitting. There are thus two entry lines per round: one for the metrics evaluated before fitting and one after fitting. Additionally, the last line represents the true labels of the test set.

For the server, the *csv* file contains the same metrics as for the clients. 
The metrics finishing by:
	- *df* represent the mean of the client metrics evaluated before fitting.
	- *d* represent the mean of the client metrics evaluated after fitting.
	- *c* represent the metric evaluated on the server testset.
Additionally, the labels predicted by the server are shown on the last line.

## Examples
Example datasets are provided in *src/databases*.

The first one is the directory *examples_scdg*. It contains graphical representation of malware. The dataset to give as parameter is *example_graph* and the model is *GINE*.

The second one is the directory *example_images*. It contains image representations of malware. The dataset to give as parameter is *example_images* and the model is *images*.


## References
\[1\] Ma, Jing, et al. "Privacy-preserving Federated Learning based on Multi-key Homomorphic Encryption." arXiv preprint arXiv:2104.06824 (2021).
