# ECE570_Project

Prerequisites: numpy, pytorch, scikit-learn, yfinance, pandas

main2.py (training the model)
  1. Hyperparameters can be adjusted from line 9~22, with the ability to swap scalers on line 131, and the ability to swap models on line 157.
  2. The main code starts with the hyperparameters that could be chosen or tuned, including the tracked tickers and epochs/dimensions of each model.
  3. Then are the functions for downloading the individual tickers and aligning them
  4. For the main part of the script it loads and aligns all training data, then splits them 8:2 sequentially, after choosing the model, it then goes into training and predicts the results.

  *  *note that line 157~177 were modified from my prior code in assignment 3*

load_n_predict.py (inference)
  1. This script loads and predicts the estimated prices for tomorrow using previously trained models. By saving everything in the model's pickle file and importing parameters from main, as long as the input tickers are the same the model should function properly.

exp1.py (for running experiment 4.1)
  1. Functions the same as main2, but with the ability to change between price, price', price'' on line 22, with mode being how many derivatives is desired. The output for this script would be the IC of all data in the test split and predictions.

The dataset used to run the experiments are also given in this repo, to use them, drag them to the root folder. If there is no pickled data in the root, the script will download the latest data.
