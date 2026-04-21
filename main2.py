import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
import yfinance
import pickle
from pathlib import Path
import pandas as pd

# input_tickers = ["GOOGL", "AMZN", "AAPL"]
# input_tickers = ["^IXIC", "MSFT", "GOOGL", "AAPL", "AMZN", "NVDA"]
input_tickers = ["^IXIC", "^VXN", "TLT", "^TNX", "DX-Y.NYB", "XLK", "NVDA"]
# input_tickers = ["^IXIC"]
# input_tickers = ["GOOGL"]
output_tickers = ["^IXIC"]
# output_tickers = ["AMZN"]
history = 60
# hidden_size = 32 * len(input_tickers)
hidden_size = 128
epochs = 50
learning_rate = 0.002
layers=3
dropout=0.2

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VanillaNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1=torch.nn.Linear(input_size*history, hidden_size)
        self.fc2=torch.nn.Linear(hidden_size, hidden_size)
        self.fc3=torch.nn.Linear(hidden_size, output_size)
        self.dropout=torch.nn.Dropout(p=0.2)
    def forward(self, x):
        out=x.flatten(start_dim=1)
        out=self.dropout(torch.nn.functional.relu(self.fc1(out)))
        out=self.dropout(torch.nn.functional.relu(self.fc2(out)))
        out=self.dropout(torch.nn.functional.relu(self.fc2(out)))
        out=self.dropout(torch.nn.functional.relu(self.fc2(out)))
        out=self.fc3(out)
        return out

class LSTM(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm=torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers, dropout=dropout, batch_first=True)
        self.fc=torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _=self.lstm(x)
        out=self.fc(out[:, -1, :])
        return out

class GRU(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gru=torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=layers, dropout=dropout, batch_first=True)
        self.fc=torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _=self.gru(x)
        out=self.fc(out[:, -1, :])
        return out

class ResLSTM(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm=torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers, dropout=dropout, batch_first=True)
        self.fc=torch.nn.Linear(hidden_size, output_size)
        self.proj=torch.nn.Linear(input_size, hidden_size)
    def forward(self, x):
        out, _=self.lstm(x)
        out=self.fc(out[:, -1, :]+self.proj(x[:, -1, :]))
        return out

class Transformer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.encoder_layer=torch.nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=4*hidden_size, dropout=dropout, batch_first=True)
        self.transformer=torch.nn.TransformerEncoder(self.encoder_layer, num_layers=layers)
        self.fc=torch.nn.Linear(hidden_size, output_size)
        self.proj=torch.nn.Linear(input_size, hidden_size)
    def forward(self, x):
        projected=self.proj(x)
        out=self.transformer(projected)
        out=self.fc(out[:, -1, :])
        return out

class ResTransformer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.encoder_layer=torch.nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=4*hidden_size, dropout=dropout, batch_first=True)
        self.transformer=torch.nn.TransformerEncoder(self.encoder_layer, num_layers=layers)
        self.fc=torch.nn.Linear(hidden_size, output_size)
        self.proj=torch.nn.Linear(input_size, hidden_size)
    def forward(self, x):
        projected=self.proj(x)
        out=self.transformer(projected)
        out=self.fc(out[:, -1, :]+projected[:, -1, :])
        return out

def load_data(ticker):
    if Path(f"{ticker}_data.pickle").exists():
        data=pickle.load(open(f"{ticker}_data.pickle", "rb"))
    else:
        print(f"Downloading {ticker}...")
        data = yfinance.download(ticker, period="5y")
        pickle.dump(data, open(f"{ticker}_data.pickle", "wb"))
    data=pd.DataFrame(data)
    return data

def load_aligned_data(tickers):
    allTickers={}
    for ticker in tickers:
        raw=load_data(ticker)
        tickerData=raw["Close"]
        tickerData.name=ticker
        allTickers[ticker]=tickerData
    data=pd.concat(allTickers, sort=True, axis=1, join="outer")
    data.interpolate(method="time", limit_area="inside", inplace=True)
    data.dropna(inplace=True)
    data=pd.DataFrame(data)
    return data

if __name__=="__main__":
    data=load_aligned_data(input_tickers)
    output_indices=[]
    for ticker in output_tickers:
        output_indices.append(input_tickers.index(ticker))
    changes = np.diff(data.values,axis=0)
    scalers=[]
    scaledChanges = np.zeros_like(changes)
    for i in range(len(input_tickers)):
        scaler=PowerTransformer()
        scaledColumn=scaler.fit_transform(changes[:, i].reshape(-1, 1))
        scaledChanges[:, i]=scaledColumn.reshape(-1)
        scalers.append(scaler)

    x, y=[], []
    for i in range(history, len(scaledChanges)):
        x.append(scaledChanges[i-history:i])
        y.append(scaledChanges[i, output_indices]-scaledChanges[i-1, output_indices])
        # y.append(scaledChanges[i, output_indices])
    x = np.array(x)
    y = np.array(y)

    split = int(len(x)*0.8)
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    # print(x_train.shape)

    model = GRU(len(input_tickers),len(output_tickers)).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()

        loss=0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_prediction = model(x_test)
            test_loss = loss_fn(test_prediction, y_test)
            if (epoch+1)%10==0:
                print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {loss:.6f}, Test Loss: {test_loss:.6f}, Variance: {test_prediction.var().item():.6f}")

    model.eval()
    with torch.no_grad():
        test_prediction_scaled=model(x_test).cpu().numpy()
    test_actual_scaled=y_test.cpu().numpy()

    test_actual=np.zeros_like(test_actual_scaled)
    test_prediction=np.zeros_like(test_prediction_scaled)
    for i, ticker in enumerate(output_tickers):
        scaler = scalers[output_indices[i]]
        test_actual[:, i]=scaler.inverse_transform(test_actual_scaled[:, i].reshape(-1, 1)).reshape(-1)
        test_prediction[:, i]=scaler.inverse_transform(test_prediction_scaled[:, i].reshape(-1, 1)).reshape(-1)
    actual_change=np.diff(data.values, axis=0)[split+history:]
    actual_change_change=np.diff(test_actual, axis=0)
    print("\nLast 5 Predictions:")
    for col, ticker in enumerate(output_tickers):
        print(f"\n{ticker}")
        print(f"{'Actual':<12}{'Predicted':<12}{'Error':<12}{'Sign'}")
        sum=0
        for i in range(-30, 0):
            actual=actual_change_change[i, col]
            prediction=test_prediction[i, col]
            error=prediction-actual
            sum+=abs(error/actual)
            print(f"${actual:<11.2f}${prediction:<11.2f}${error:<11.2f}{(actual*prediction)>0}")
        print(f"Avg Error: {sum/30*100:.2f}%")

    pickle.dump((model, scalers, input_tickers, output_tickers), open(f"{"_".join(input_tickers)}_multi_model.pickle", "wb"))