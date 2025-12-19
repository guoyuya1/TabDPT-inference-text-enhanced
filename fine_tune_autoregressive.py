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


def preprocess_data():
    # Load climate dataframe
    df = pd.read_csv("../../MultimodalForcast/data/climate_ttc/climate_2014_2023_final_with_embeddings_lag_3.csv")

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

def freeze_all_except_alpha(model):
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze alpha parameters
    for layer in model.transformer_encoder:
        if hasattr(layer, 'alpha'):
            layer.alpha.requires_grad = True

def main():
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    # Initialize regressor with text_enhanced=True
    print("Initializing TabDPTRegressor...")
    regressor = TabDPTRegressor(text_enhanced=True, device=device)
    regressor.fit(X_train, y_train, text=train_text)
    
    # Extract model
    model = regressor.model
    model.train() 

    freeze_all_except_alpha(model)

    x_src, y_src, text_enhanced_attn_weight = regressor._predict_autoregressive_fine_tune(X_test, text=test_text)

    pred = model(
    x_src=torch.cat([X_train, X_test], dim=1),
    y_src=y_train.unsqueeze(-1),
    task="reg",
    text_enhanced_attn_weight=text_enhanced_attn_weight)

    loss = nn.MSELoss()(pred, y_src)



