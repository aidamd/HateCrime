# Event Detection

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

The following libraries are only used in the crawling script. You should download them of you are going to use the code in Crawl directory:
```
bs4 4.6.0
splinter 0.7.7
newspaper3k 0.2.6
```

## Running the code

All the parameters of the code are denoted in params.json.
The parameters are defined as following:

```
  "task": "detect and extract" # detect and extract are the two tasks
  "num_layers": 1 # number of LSTM layers
  "hidden_size": 256 # hidden size of the LSTM
  "art_filter_sizes": [2, 3, 4] # filter sizes in detect model
  "art_num_filters": 10 # number of different filters in detect model
  "pretrain": true # if set to true, uses Glove embeddings
  "embedding_size": 300 # size of the embedding
  "learning_rate": 0.00001 # learning rate for the detect task
  "keep_ratio": 0.75 # keep ratio for the detect task
  "epochs": 30 # number of epochs
  "cell": "GRU" # cells used in RNN
  "entity_keep_ratio": 0.75 # keep ratio in extract task
  "entity_learning_rate": 0.001 # learning rate in extract task
  "entity_num_filters": 3 # number of filters in extract task
  "batch_size": 10 # size of batches, shows the number of articles in each batch
```

### Testing the code

In order to test the detect task, make sure that the "task" parameter in param.json includes the word "detect" and run the following:

```
python3 run.py
```

If the "task" parameter also include "extract", the extract task will be tested too.


