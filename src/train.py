from sklearn.preprocessing import MinMaxScaler
import torch
from src.model import *
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def train(model, train_loader, val_loader, config, device):
    lr = config['model']['learning_rate']
    epochs = config['model']['epochs']
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v_inputs, v_labels in val_loader:
                v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
                
                v_outputs = model(v_inputs)
                v_loss = criterion(v_outputs, v_labels)
                val_loss += v_loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        train_loss_history.append(avg_train)
        val_loss_history.append(avg_val)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"--> Strategy: Model saved with Val Loss: {best_val_loss:.6f}")
            

    print("Training complete. Loading best model...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, train_loss_history, val_loss_history






