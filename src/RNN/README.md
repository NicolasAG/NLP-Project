# LSTM RNN model

First, you need to create a few directories:

- where prediction files (.csv) will be saved: `mkdir ModelResponses`
- where trained models will be saved: `mkdir LSTM-Models`

Then build some predictions using `python lstm.py <dataset>`

with the following flags:

- -v  activate verbose: increase output verbosity (default=inactive).
- `<dataset>`  must be either 'imdb' or 'rottom'.
- --feature_size  embedding size = number of hidden LSTM units = features per review (default=300).
- --extend_data  either 'true' or 'false' to know if using the extended version of the dataset (default=False).
- --optimizer  either 'adam' or 'adadelta' or 'rmsprop': optimiser algorithm to do parameter updates.
- ... more of them in `lstm.py` file

Model described in http://deeplearning.net/tutorial/lstm.html

