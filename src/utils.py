from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
import os 

def load_config(config_path: str="config.yaml") -> dict:
    '''
    Helper function that loads the config.yaml into a dictonary.
    
    :param config_path: The path to the config file. 
    :type config_path: str
    :return: Returns the yaml file as a dictonary
    :rtype: dict
    '''

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    '''
    Returns a dictonary with useful metrics in determining the quality of the model.

    :param y_true: A numpy array of all the true data
    :type y_true: np.ndarray
    :param y_pred: A numpy array of the predictions the model came up with
    :type y_pred: np.ndarray
    :return: Returns the metrics to determine the quality of the model
    :rtype: dict
    '''
    # The smaller the better.
    mse = mean_squared_error(y_true, y_pred)
    # RMSE is more interpetable than MSE so we'll use that
    # Smaller the better
    rmse = np.sqrt(mse)
    
    # Error in terms of percentage, again smaller the better.
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # How closely the model follows the trend
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def plot_stock(df: pd.DataFrame) -> None:
    """
    A quick way to plot stocks recieved by get_ticker_data.
    
    :param df: Dataframe from get_ticker_data
    :type df: pd.DataFrame
    """
    ticker = df.columns.name 
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df.index, df['Close'], label = 'Close')
    ax1.set_title(f'{ticker} Close', fontsize=16)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_stock_technicals(df: pd.DataFrame) -> None:
    """
    Plots the tehcnical indicators from the DF, still work in progress
    
    :param df: Data from get_ticker_data
    :type df: pd.DataFrame
    """
    ticker = df.columns.name
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True, 
                                            gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Close and Bollinger Bands
    ax1.plot(df.index, df['Close'], label='Close', color='cyan', alpha=0.8)
    ax1.plot(df.index, df['BB_Upper'], label='Upper Band', color='white', linestyle='--', alpha=0.3)
    ax1.plot(df.index, df['BB_Lower'], label='Lower Band', color='white', linestyle='--', alpha=0.3)
    ax1.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], color='gray', alpha=0.1)
    ax1.set_title(f'{ticker} Technical Analysis', fontsize=16)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.2)

    # RSI 
    ax2.plot(df.index, df['RSI'], color='magenta', label='RSI')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5) 
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5) 
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI')
    ax2.grid(alpha=0.2)

    # MACD
    ax3.plot(df.index, df['MACD'], color='dodgerblue', label='MACD')
    ax3.plot(df.index, df['MACD_Signal'], color='orange', label='Signal')
    colors = ['green' if x > 0 else 'red' for x in df['MACD_Hist']]
    ax3.bar(df.index, df['MACD_Hist'], color=colors, alpha=0.5, label='Hist')
    ax3.set_ylabel('MACD')
    ax3.legend(loc='upper left')
    ax3.grid(alpha=0.2)

    # KDJ
    ax4.plot(df.index, df['KDJ_K'], color='white', label='K', alpha=0.7)
    ax4.plot(df.index, df['KDJ_D'], color='yellow', label='D', alpha=0.7)
    ax4.plot(df.index, df['KDJ_J'], color='purple', label='J', alpha=0.9)

    # KDJ levels 
    ax4.axhline(80, color='red', linestyle=':', alpha=0.4)
    ax4.axhline(20, color='green', linestyle=':', alpha=0.4)
    ax4.set_ylabel('KDJ')
    ax4.legend(loc='upper left')
    ax4.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

def evaluate_model(test_loader, scaler, model, device):
    checkpoint_path = 'best_model.pth' # do more research on pth files
    
    # load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Checkpoint not found, using current model weights.")

    model.eval() # Because we have dropout function
    
    predictions = []
    actuals = []

    # Run predictions
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(labels.numpy())

    predictions = np.concatenate(predictions).reshape(-1, 1)
    actuals = np.concatenate(actuals).reshape(-1, 1)

    # --- FIX 2: Dynamic Inverse Scaling ---
    # We dynamically get the number of features the scaler was trained on
    # instead of hardcoding '10'.
    num_features = scaler.n_features_in_ 
    
    # Create dummy arrays with the correct shape
    dummy_pred = np.zeros((len(predictions), num_features))
    dummy_actual = np.zeros((len(actuals), num_features))
    
    # We assume 'Close' price is at index 3 (Open, High, Low, Close...)
    # If your column order changed, you might need to update this index.
    target_col_idx = 3 
    
    dummy_pred[:, target_col_idx] = predictions.flatten()
    dummy_actual[:, target_col_idx] = actuals.flatten()

    # Inverse Transform to get actual Dollars
    inv_predictions = scaler.inverse_transform(dummy_pred)[:, target_col_idx]
    inv_actuals = scaler.inverse_transform(dummy_actual)[:, target_col_idx]

    # Calculate Metrics
    metrics = calculate_metrics(inv_actuals, inv_predictions)
    
    print(f"Final Test Metrics:")
    print(f"RMSE: ${metrics['RMSE']:.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"R^2:  {metrics['R2']:.4f}")

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(inv_actuals, label='Actual Price', color='cyan', alpha=0.7)
    plt.plot(inv_predictions, label='Predicted Price', color='orange', linestyle='--', linewidth=1.5)
    plt.title(f"Forecast Results | RMSE: ${metrics['RMSE']:.2f} | MAPE: {metrics['MAPE']:.2f}% | R²: {metrics['R2']:.4f}", fontsize=14)
    plt.xlabel('Days (Test Set)')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

def plot_loss_curve(train_losses, val_losses):
    """
    Plots the training and validation loss curves.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()