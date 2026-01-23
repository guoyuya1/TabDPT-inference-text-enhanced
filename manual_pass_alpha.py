import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tabdpt import TabDPTRegressor


def extract_and_stack_embeddings(df, embedding_cols):
    """
    Extract embedding columns from dataframe and stack them as a numpy matrix.
    
    Args:
        df: DataFrame with embedding columns stored as string representations of lists
        embedding_cols: List of column names to extract (e.g., ['embedding_text_lag1', 'embedding_text_lag2', 'embedding_text_lag3'])
    
    Returns:
        numpy array of shape (N, L, D) where:
        - N = number of rows
        - L = number of embedding columns (lags)
        - D = embedding dimension
    """
    embeddings_list = []
    
    for col in embedding_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Skipping.")
            continue
        
        # Parse string representation of list to actual list
        col_embeddings = []
        for idx, val in enumerate(df[col]):
            try:
                # Try to parse as literal string representation
                if isinstance(val, str):
                    embedding = ast.literal_eval(val)
                else:
                    embedding = val
                col_embeddings.append(np.array(embedding))
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse embedding at row {idx} in column {col}: {e}")
                # Use zeros as fallback (you may want to handle this differently)
                if len(col_embeddings) > 0:
                    col_embeddings.append(np.zeros_like(col_embeddings[0]))
                else:
                    # If we don't know the dimension yet, skip this row
                    continue
        
        embeddings_list.append(np.array(col_embeddings))
    
    # Stack along the lag dimension: (N, L, D)
    if embeddings_list:
        stacked = np.stack(embeddings_list, axis=1)  # (N, L, D)
        return stacked
    else:
        raise ValueError("No valid embedding columns found")


def preprocess_data_climate():
    # Load climate dataframe
    df = pd.read_csv("../MultimodalForcast/data/climate_ttc/climate_2014_2023_final_with_embeddings_lag_3.csv")

    # Sort by date if available
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    # Configuration: Select which columns to use as features
    feature_columns = ["precip_lag1", "precip_lag2", "precip_lag3",
                    "humidity_lag1", "humidity_lag2", "humidity_lag3",
                    "windspeed_lag1", "windspeed_lag2", "windspeed_lag3",
                    "temp_lag1", "temp_lag2", "temp_lag3"]


    # Validate and extract features
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following columns are not found in the dataframe: {missing_cols}")
        feature_columns = [col for col in feature_columns if col in df.columns]

    X = df[feature_columns].values
    # target column
    y = df["temp"].values

    print(f"Using {len(feature_columns)} feature columns: {feature_columns}")

    # Extract text embeddings for all data first
    embedding_cols = ['embedding_text_lag1', 'embedding_text_lag2', 'embedding_text_lag3']
    text_embeddings_all = extract_and_stack_embeddings(df, embedding_cols)

    # Split based on date (time series split)
    split_ratio = 0.8
    split_idx = int(len(df) * split_ratio)

    # Alternatively, you can use a specific date:
    # split_date = "2022-12-05"
    # if "date" in df.columns:
    #     df['date'] = pd.to_datetime(df['date'])
    #     split_idx = len(df[df['date'] < pd.to_datetime(split_date)])

    idx_train = np.arange(split_idx)
    idx_test = np.arange(split_idx, len(df))

    # Split data using date-based indices
    X_train = X[idx_train]
    X_test = X[idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]

    # Split text embeddings using the same indices
    train_text = text_embeddings_all[idx_train]
    test_text = text_embeddings_all[idx_test]

    # Print split information
    if "date" in df.columns:
        print(f"Train date range: {df.iloc[idx_train[0]]['date']} to {df.iloc[idx_train[-1]]['date']}")
        print(f"Test date range: {df.iloc[idx_test[0]]['date']} to {df.iloc[idx_test[-1]]['date']}")

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"train_text shape: {train_text.shape}")
    print(f"test_text shape: {test_text.shape}")

    return X_train, X_test, y_train, y_test, train_text, test_text


def preprocess_data_bitcoin():
    # Load bitcoin dataframe
    df = pd.read_csv("/home/mzzhang/tabdpt/data/bitcoin/bitcoin_final_with_embeddings_lag_3.csv").head(1000)

    # Sort by date if available
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    # Configuration: Select which columns to use as features
    feature_columns = ["Open_lag1", "Open_lag2", "Open_lag3",
                    "High_lag1", "High_lag2", "High_lag3",
                    "Low_lag1", "Low_lag2", "Low_lag3",
                    "Close_lag1", "Close_lag2", "Close_lag3",
                    "Adj_Close_lag1", "Adj_Close_lag2", "Adj_Close_lag3"]

    # Validate and extract features
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following columns are not found in the dataframe: {missing_cols}")
        feature_columns = [col for col in feature_columns if col in df.columns]

    X = df[feature_columns].values
    # target column
    y = df["Adj_Close"].values

    print(f"Using {len(feature_columns)} feature columns: {feature_columns}")

    # Extract text embeddings for all data first
    embedding_cols = ['embedding_summary_gpt-5-mini_lag1', 'embedding_summary_gpt-5-mini_lag2', 'embedding_summary_gpt-5-mini_lag3']

    text_embeddings_all = extract_and_stack_embeddings(df, embedding_cols)

    # Split based on date (time series split)
    split_ratio = 0.763
    split_idx = int(len(df) * split_ratio)

    # Alternatively, you can use a specific date:
    # split_date = "2022-12-05"
    # if "date" in df.columns:
    #     df['date'] = pd.to_datetime(df['date'])
    #     split_idx = len(df[df['date'] < pd.to_datetime(split_date)])

    idx_train = np.arange(split_idx)
    idx_test = np.arange(split_idx, len(df))

    # Split data using date-based indices
    X_train = X[idx_train]
    X_test = X[idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]

    # Split text embeddings using the same indices
    train_text = text_embeddings_all[idx_train]
    test_text = text_embeddings_all[idx_test]

    # Print split information
    if "date" in df.columns:
        print(f"Train date range: {df.iloc[idx_train[0]]['date']} to {df.iloc[idx_train[-1]]['date']}")
        print(f"Test date range: {df.iloc[idx_test[0]]['date']} to {df.iloc[idx_test[-1]]['date']}")

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"train_text shape: {train_text.shape}")
    print(f"test_text shape: {test_text.shape}")

    return X_train, X_test, y_train, y_test, train_text, test_text

def get_alpha_parameters(model):
    """Extract alpha parameters from the model."""
    alpha_params = []
    for layer in model.transformer_encoder:
        if hasattr(layer, 'alpha'):
            alpha_params.append(layer.alpha)
    return alpha_params

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Preprocess data and time series split
    X_train, X_test, y_train, y_test, train_text, test_text = preprocess_data_bitcoin()

    N = 1
    X_test = X_test[:N]
    y_test = y_test[:N]
    test_text = test_text[:N]
    
    # Initialize regressor with text_enhanced=True
    print("Initializing TabDPTRegressor...")
    regressor = TabDPTRegressor(text_enhanced=True, device=device)
    regressor.fit(X_train, y_train, text=train_text)
    
    # Extract model
    dpt_model = regressor.model
    dpt_model.eval()
    
    # Get alpha parameters
    alpha_params = get_alpha_parameters(dpt_model)
    if len(alpha_params) == 0:
        raise ValueError("No alpha parameters found! Make sure text_enhanced=True when initializing TabDPTRegressor.")
    print(f"Found {len(alpha_params)} alpha parameter(s)")
    
    # Prepare data for forward pass
    X_train_tensor, X_test_tensor, y_train_tensor, text_enhanced_attn_weight = regressor._predict_autoregressive_fine_tune(
        X_test, text=test_text
    )
    
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Test gating values from 0 to 1, convert to raw alpha using inverse sigmoid
    print(f"\nTesting gating values (sigmoid of alpha) from 0.0 to 1.0 in steps of 0.1...")
    print(f"{'Gating':<12} {'Alpha (raw)':<18} {'MSE Loss':<12} {'MAE Loss':<12} {'RMSE':<12}")
    print("-" * 60)
    
    results = []
    gating_values_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for gating_val in gating_values_to_test:
        # Convert gating value to raw alpha using inverse sigmoid (logit)
        # Handle edge cases: gating=0 -> alpha=-inf, gating=1 -> alpha=+inf
        if gating_val == 0.0:
            alpha_raw = -10.0  # Approximate -inf
        elif gating_val == 1.0:
            alpha_raw = 10.0   # Approximate +inf
        else:
            alpha_raw = np.log(gating_val / (1 - gating_val))  # Inverse sigmoid
        
        # Set all alpha parameters to the calculated raw value
        with torch.no_grad():
            for alpha_param in alpha_params:
                alpha_param.fill_(alpha_raw)
        
        # Forward pass with current alpha values
        with torch.no_grad():
            preds = dpt_model(
                x_src=torch.cat([X_train_tensor, X_test_tensor], dim=1),
                y_src=y_train_tensor.unsqueeze(-1),
                task="reg",
                text_enhanced_attn_weight=text_enhanced_attn_weight
            )
            preds = preds.squeeze(-1)
            
            # Calculate MSE loss on test set
            mse_loss = torch.nn.functional.mse_loss(preds, y_test_tensor)
            # Calculate MAE (Mean Absolute Error)
            mae_loss = torch.nn.functional.l1_loss(preds, y_test_tensor)
            # Calculate RMSE (Root Mean Squared Error)
            rmse = torch.sqrt(mse_loss)
        
        # Store results
        results.append({
            'gating': gating_val,
            'alpha_raw': alpha_raw,
            'mae_loss': mae_loss.item(),
            'mse_loss': mse_loss.item(),
            'rmse': rmse.item()
        })

        print(f"{gating_val:<12.1f} {alpha_raw:<18.6f} {mse_loss.item():<12.4f} {mae_loss.item():<12.4f} {rmse.item():<12.4f}")
    
    # Find best gating value (based on MSE loss)
    best_result = min(results, key=lambda x: x['mse_loss'])
    print(f"\nBest gating value: {best_result['gating']:.1f}")
    print(f"Best alpha (raw): {best_result['alpha_raw']:.6f}")
    print(f"Best MSE loss: {best_result['mse_loss']:.4f}")
    print(f"Best MAE loss: {best_result['mae_loss']:.4f}")
    print(f"Best RMSE: {best_result['rmse']:.4f}")



if __name__ == "__main__":
    main()
