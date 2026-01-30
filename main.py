import torch
from src.utils import load_config, evaluate_model, plot_loss_curve
from src.dataset import get_ticker_data, create_dataloaders
from src.model import HybridCNNLSTM
from src.train import train

def main():
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(f"Configuration loaded for: {config['data']['target_ticker']}")

    ticker = config['data']['target_ticker']
    df = get_ticker_data(ticker, config)
    
    if df is None or df.empty:
        print("Error: Data not found or empty.")
        return
    
    # Make it work with any amount of features. 
    features_to_use = [
        'Open', 'High', 'Low', 'Close', 'Log_Returns', 
        'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'SMA_50'
    ]
    df = df[features_to_use]
    train_loader, val_loader, test_loader, scaler = create_dataloaders(df, config)
    print(f"Data processed. Training batches: {len(train_loader)}")

    model = HybridCNNLSTM().to(device)

    print("Starting training")
    model, train_losses, val_losses = train(model, train_loader, val_loader, config, device)
    plot_loss_curve(train_losses, val_losses)
    print("Evaluating best model on Test Set")
    evaluate_model(test_loader, scaler, model, device)

if __name__ == "__main__":
    main()



