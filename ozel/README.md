# Set up
To run experiments, ensure a python environment with numpy, pandas, scikit-learn, tqdm, matplotlib.

Place your data in the root directory under data directory. Your path to data should be like `./data/credit_approval.csv`, `./data/parkinsons.csv`, and `./data/rice.csv`.

# Running experiments
The are 4 main python files to run each dataset experiment. Each experiment will save outputs under `./results/` and `./figures/`.

## MNIST Dataset
Run `python main_mnist.py` to run the experiment. The outputs will start with `mnist_*`.

## Credit Approval Dataset
Run `python main_credit.py` to run the experiment. The outputs will start with `credit_*`.

## Parkinsons Dataset
Run `python main_parkinsons.py` to run the experiment. The outputs will start with `parkinsons_*`.

## Rice Dataset
Run `python main_rice.py` to run the experiment. The outputs will start with `rice_*`.
