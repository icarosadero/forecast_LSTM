import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import pandas as pd
import numpy as np
import sqlite3 as sq
from tqdm import tqdm

"""
Constants
"""
t_ = "TIMESTAMP"
tn_ = "T_n"
T_ = "T"
a = 0.2

sigmoid = lambda x: 1/(1 + np.exp(-x))
inv_sigmoid = lambda x: np.log(x/(1-x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=5, output_size=1, window=10):
        super().__init__()
        self.window = window
        
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).double(),
                            torch.zeros(1,1,self.hidden_layer_size).double())
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        inpt = input_seq.view(len(input_seq) ,1, -1).double()
        lstm_out, self.hidden_cell = self.lstm(inpt, self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return self.sigmoid(predictions[-self.window:].view(-1))
            

def get_t(df,dt_min,dt_max=None):
    if dt_max:
        return df.loc[(df[t_]>=dt_min) & (df[t_]<dt_max)]
    else:
        return df.loc[df[t_]>=dt_min]

def window_iter(sequence, window):
    for i in range(0,len(sequence)-window+1):
        yield sequence[i:i+window]    

def sampler(sequence, timestamps, window=10, timedelta=datetime.timedelta(hours=1,minutes=30)):
    dregions = []#Discontinuous regions
    chunk = []
    for i in range(len(timestamps)-1):
        if timestamps[i+1]-timestamps[i]<timedelta:
            chunk.append(sequence[i])
        else:
            dregions.append(chunk[:])
            chunk = []
    dregions = [chunk] if len(dregions)==0 else dregions
    for d in dregions:
        return window_iter(d,window)

def timestamp_fill(df,t=t_,dt=datetime.timedelta(hours=1,minutes=30),freq="1H"):
    mask = df[t].diff() > dt
    for j in df[mask].index:
        if j>1:
            data = pd.date_range(start=df.loc[j-1,t],end=df.loc[j,t],freq=freq)
            data = data.to_frame().reset_index(drop=True)
            data.columns = [t]
            df = pd.concat([df,data],ignore_index=True)
    df = df.sort_values(by=t)
    df = df.reset_index(drop=True)
    return df

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    """
    Dataset preparation
    """
    df = pd.DataFrame()
    with sq.connect("database.db") as db:
        df = pd.read_sql("select T,TIMESTAMP from bmp order by ID DESC limit 2048",db,parse_dates={'TIMESTAMP':{'infer_datetime_format':True}})
    df = df.iloc[::-1]
    df = df.reset_index(drop=True)
    df = timestamp_fill(df)
    df = df.fillna(0)
    df["T_n"] = sigmoid(a*(df[T_]-25))
    
    """
    Parameters
    """
    window = 48 #Window subsequence sampling size. Must be even.
    assert window%2==0
    b = 12 #Split test and training data size
    timestamp = df[t_]
    temp_norm = df[tn_]
    dt_min, dt_max = timestamp.min(), False
    
    """
    Data
    """
    data = np.array(list(sampler(temp_norm.tolist(),timestamp.tolist(),window=window)))
    
    ipt = torch.from_numpy(data[:,:-b])
    target = torch.from_numpy(data[:,-b:])
    #ipt = torch.from_numpy(data[:,:-1])
    #target = torch.from_numpy(data[:,1:])
    
    test_input_df = get_t(df,dt_min,dt_max)
    test_input = torch.from_numpy(test_input_df[tn_].to_numpy())
    test_input = torch.from_numpy(df["T_n"].to_numpy()[-window:-b]).view(-1)
    x = [test_input_df[t_].max() + datetime.timedelta(hours=i) for i in range(b)]

    """
    Model training
    """
    #model = LSTM()
    model = LSTM(window=b)
    model = model.double()
    model = model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 250
    for i in tqdm(range(epochs)):
        for seq, labels in zip(ipt,target):
            seq = seq.double().to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).double().to(device),
                            torch.zeros(1, 1, model.hidden_layer_size).double().to(device))
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i%5 == 0:
            tqdm.write(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    
    """
    Model prediction
    """
    model.eval()
    seq = test_input

    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).double().to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).double().to(device))
        y = model(seq.double().to(device)).detach().cpu().numpy()
    assert len(x)==len(y)    
    
    """
    Plotting
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df[T_].to_numpy(),
        x=timestamp,
        mode='lines+markers',
        name='data'))
    fig.add_trace(go.Scatter(
        y=inv_sigmoid(y)/a+25.,
        x=x,
        mode='lines+markers',
        name='prediction'))
    fig.add_trace(go.Scatter(
        y=test_input_df[T_].to_numpy(),
        x=test_input_df[t_].tolist(),
        mode='lines+markers',
        name='input'))

    fig.write_html("predict.html")
