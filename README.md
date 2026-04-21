# ECE570_Project

Prerequisites: numpy, pytorch, sklearn, yfinance, pandas

main2.py (training the model)
  1. The main code starts with the hyperparameters that could be chosen or tuned, including the tracked tickers and epochs/dimensions of each model.
  2. Then are the functions for downloading the individual tickers and aligning them
  3. For the main part of the script it loads and aligns all training data, then splits them 8:2 sequentially, after choosing the model, it then goes into training and predicts the results.
  
  *  *note that line 117~133 were modified from my prior code in assignment 3*

load_n_predict.py (inference)
  1. This script loads and predicts the estimated prices for tomorrow using previously trained models. By saving everything in the model's pickle file and importing parameters from main, as long as the input tickers are the same the model should function properly.
