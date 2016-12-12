# Word to Vec features

First, you need to create a few directories:

- where prediction files (.csv) will be saved: `mkdir ModelResponses`
- where W2V models will be saved: `mkdir W2VModels`
- where W2V clusters will be saved: `mkdir W2VClusters`

Two versions of features:

- average word embeddings to get review embedding (use `random_forest.py`)
- cluster word vectors and build bag-of-centroids features (use `random_forest-KMEAN.py`)

Then build some predictions using `python <script>.py <dataset>`

with the following flags:

- `<dataset>`  must be either 'imdb' or 'rottom'.
- --feature_size  embedding size = number of features per review (default=300).
- --forest_size  number of decision trees in the random forest algorithm (default=100).
- --extend  either 'true' or 'false' to know if using the extended version of the dataset (default=False).

