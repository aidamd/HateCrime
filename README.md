# Hate crime detection and entity extraction

This project is composed of two models to detect and extract hate crime events from news articles.

## Getting Started

In order to run the code you need to download the Data and embeddings directory.

### Prerequisites

This project uses Python 3.6.2. The following libraries must be installed:

```
sklearn 0.19.1
tensorflow-gpu 1.11.0
pandas 0.23.0
nltk 3.2.5
tqdm 4.28.1
numpy 1.14.3

```

## Parameters

All the parameters of the code are denoted in params.json.
The parameters are defined as following:

```
  "hidden_size": 100 # hidden size of the LSTM
  "art_filter_sizes": [2, 3, 4] # filter sizes in detect model
  "art_num_filters": 10 # number of different filters in detect model
  "pretrain": true # if set to true, uses Glove embeddings
  "embedding_size": 300 # size of the embedding
  "learning_rate": 0.00001 # learning rate for the detect task
  "keep_ratio": 0.75 # keep ratio for the detect task
  "epochs": 30 # number of epochs
  "entity_keep_ratio": 0.75 # keep ratio in extract task
  "entity_learning_rate": 0.001 # learning rate in extract task
  "batch_size": 5 # size of batches, shows the number of articles in each batch
```

### Running the detection code

In order to run the detection code, use the following script:

`python3 run_detect.py --model <MODEL_NAME> --goal <GOAL> --dataset <DATASET> --params <PARAMS_FILE>`

substitude the following tokens according to the task in mind:

- `<MODEL_NAME>`: you can either use `MICNN` (the model used in the paper) or `ATTN` (the hierarchical attention baseline)
- `<GOAL>`: the goal of the task is either `train` or `predicts`
- `<DATASET>`: use one of the three datasets (`hate`, `kidnap` or `homicide`) to perform the detection 
- `<PARAM_FILE>`: is the .json file that includes all the model parameters. The model uses `params.json` as default.


### Running the extraction code

In order to run the detection code, use the following script:

`python3 run_extract.py --goal <GOAL> --params <PARAMS_FILE>`

substitude the following tokens according to the task in mind:

- `<GOAL>`: the goal of the task is either `train` or `predicts`
- `<PARAM_FILE>`: is the .json file that includes all the model parameters. The model uses `params.json` as default.




```
python3 run.py
```

If the "task" parameter also include "extract", the extract task will be tested too.


