# DocNADE

A TensorFlow implementation of the DocNADE model, published in [A Neural Autoregressive Topic Model](https://papers.nips.cc/paper/4613-a-neural-autoregressive-topic-model).


## Requirements

Requires Python 3 (tested with `3.6.1`). The remaining dependencies can then be installed via:

        $ pip install -r requirements.txt
        $ python -c "import nltk; nltk.download('punkt')"


## Data format and preprocessing

You first need to preprocess any input data into the format expected by the model:

        $ python preprocess.py --input <path to input dataset> --output <path to output dataset> --vocab <path to vocab file>

where
`<path to input directory>` points to a directory containing an input dataset (described below),
`<path to output directory>` gives the path to a newly created output dataset directory (containing the preprocessed data), and
`<path to vocab file>` gives the path to a vocabulary file (described below).

**Datasets**: A directory containing CSV files. There is expected to be 1 CSV file per set or collection, with separate sets for training, validation and test. The CSV files in the directory must be named accordingly: `training.csv`, `validation.csv`, `test.csv`. For this task, each CSV file (prior to preprocessing) consists of 2 string fields with a comma delimiter - the first is the label and the second is the document body.

**Vocabulary files**: A plain text file, with 1 vocabulary token per line (note that this must be created in advance, we do not provide a script for creating vocabularies). We do provide the vocabulary file used in our 20 Newsgroups experiment in [`data/20newsgroups.vocab`](data/20newsgroups.vocab).


## Training

The default parameters should achieve good perplexity results, you just need to pass the input dataset and model output directories:

        $ python train.py --dataset <path to preprocessed dataset> --model <path to model output directory>

To view additional parameters (which may yield better document representations):

        $ python train.py --help


## Extracting document vectors and evaluating results

To evaluate the retrieval results:

        $ python evaluate.py --dataset <path to preprocessed dataset> --model <path to trained model directory>

To extract document vectors (will be saved in NumPy text format to the model directory):

        $ python vectors.py --dataset <path to preprocessed dataset> --model <path to trained model directory>
