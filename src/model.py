import torch
import torch.nn as nn

class HybridCNNLSTM(torch.nn.Module):
    def __init__(self):
        super(HybridCNNLSTM, self).__init__()
        self.conv1 = torch.nn.Conv1d(10, 128, kernel_size=3)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=3)
        self.lstm1 = torch.nn.LSTM(128, 100, batch_first=True)
        self.dropout1 = torch.nn.Dropout(p=0.2)
        self.lstm2 = torch.nn.LSTM(input_size=100, hidden_size=100, batch_first=True)
        self.dropout2 = torch.nn.Dropout(p=0.2)
        self.dense1 = torch.nn.Linear(100, 50)
        self.dense2 = torch.nn.Linear(50, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        
        x = self.pool1(x)
        
        # CNN/Pool output is (Batch, 128, 1)
        # LSTM needs (Batch, 1, 128)
        x = x.permute(0, 2, 1) 
        
        x, (hn, cn) = self.lstm1(x)
        x = self.dropout1(x)
        
        x, (hn2, cn2) = self.lstm2(x)
        x = self.dropout2(x)
        
        #x = x.squeeze(1)
        x = x[:, -1, :]
        x = torch.relu(self.dense1(x))
        x = self.dense2(x)
        
        return x


