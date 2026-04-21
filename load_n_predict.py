import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
import yfinance
import pickle
from pathlib import Path
from main2 import *

tickers=["^IXIC", "^VXN", "TLT", "^TNX", "DX-Y.NYB", "XLK", "NVDA"]

model, scalers, inputTickers, outputTickers=pickle.load(open(f"{"_".join(tickers)}_multi_model.pickle", "rb"))
model.to("cuda")
output_indices=[inputTickers.index(t) for t in outputTickers]
data=load_aligned_data(inputTickers)
prices=data.values
changes=np.diff(prices, axis=0)

scaledChanges=np.zeros_like(changes)
for i in range(len(inputTickers)):
    scaledChanges[:, i]=scalers[i].transform(changes[:, [i]]).reshape(-1)
test_sequence=torch.FloatTensor(scaledChanges[-history:].reshape(1, history, len(inputTickers))).to("cuda")

model.eval()
with torch.no_grad():
    test_prediction_scaled = model(test_sequence).cpu().numpy()
test_prediction = np.zeros_like(test_prediction_scaled)
for i, ticker in enumerate(output_tickers):
    scaler = scalers[output_indices[i]]
    test_prediction[:, i] = scaler.inverse_transform(test_prediction_scaled[:, i].reshape(-1, 1)).reshape(-1)
    print(f"Today's Price:               ${prices[-1][i]:.2f}")
    print(f"Predicted Change:            ${test_prediction[-1][i]+changes[-1][i]:.2f}")
    print(f"Tomorrow's Predicted Price:  ${prices[-1][i]+test_prediction[-1][i]+changes[-1][i]:.2f}")