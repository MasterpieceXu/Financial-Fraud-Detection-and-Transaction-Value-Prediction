import time
import pandas as pd
import numpy as np
from datetime import datetime

# --- 1. Import All Required Sklearn Modules ---
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor # Our chosen model

# --- 2. Define Helper Functions ---

def get_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two points on Earth.
    """
    R = 6371  # Earth's radius in km
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance

def preprocess(df):
    """
    Applies all cleaning and feature engineering steps.
    (This is the perfect function you defined in your notebook)
    """
    df = df.copy()
    
    # 1. Time Features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['is_weekend'] = df['trans_day_of_week'].isin([5, 6]).astype(int)

    # 2. Age Feature
    df['dob'] = pd.to_datetime(df['dob'])
    snapshot_date = datetime(2021, 1, 1) # (Use a fixed snapshot date for consistency)
    df['age'] = ((snapshot_date - df['dob']).dt.days / 365.25).astype(int)
    
    # 3. Distance Feature
    df['customer_merchant_distance'] = get_distance(
        df['lat'], df['long'],
        df['merch_lat'], df['merch_long']
    )

    # 4. Merchant Feature
    df['merchant_has_fraud_prefix'] = df['merchant'].str.startswith('fraud_').astype(int)

    # 5. Fill Missing Values (safer to do it in preprocess)
    df['customer_merchant_distance'] = df['customer_merchant_distance'].fillna(df['customer_merchant_distance'].median())
    df['age'] = df['age'].fillna(df['age'].median())

    # 6. Final Column Selection
    columns_to_keep = [
        'amt',                    # Target 1 (Regression)
        'is_fraud',               # Target 2 (Classification)
        'trans_num',              # ID
        # --- Features ---
        'category',
        'gender',
        'city_pop',
        'trans_hour',
        'trans_day_of_week',
        'is_weekend',
        'age',
        'customer_merchant_distance',
        'merchant_has_fraud_prefix'
    ]
    
    # Ensure all columns exist, even if empty (for test.csv)
    final_cols = [col for col in columns_to_keep if col in df.columns]
    return df[final_cols]

# --- 3. Define the Final Preprocessor ---

# These lists MUST match the output of the preprocess() function
numeric_features = [
    'city_pop', 'trans_hour', 'trans_day_of_week', 'is_weekend', 
    'age', 'customer_merchant_distance', 'merchant_has_fraud_prefix'
]
categorical_features = [
    'category', 'gender'
]

# Numeric Transformer
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Double-check
    ('scaler', StandardScaler())
])

# Categorical Transformer
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Double-check
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # Must ignore unknown categories
])

# Combine into the final preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# --- 4. Define the Final Regression Model Pipeline ---

regression_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', HistGradientBoostingRegressor(random_state=42)) # Use defaults - fast and accurate
])

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    
    print("--- Part II: Regression Script Starting ---")
    start_time = time.time()

    # 1. Load Data (Full Datasets!)
    print("Loading train.csv (1M rows) and test.csv...")
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
    except FileNotFoundError:
        print("Error: train.csv or test.csv not found. Please ensure they are in the same directory.")
        exit()
        
    # Keep test IDs for the final submission
    test_ids = test_df['trans_num']

    # 2. Preprocess Data
    print("Preprocessing training and test sets...")
    processed_train_df = preprocess(train_df)
    processed_test_df = preprocess(test_df) # Apply identical steps to test set

    # 3. Prepare Training Data (X_train, y_train)
    print("Preparing X_train and y_train...")
    # !! CRITICAL: We are predicting the log of amt
    X_train = processed_train_df.drop(columns=['amt', 'is_fraud', 'trans_num'])
    y_train = np.log1p(processed_train_df['amt']) # log1p = log(1 + x)

    # 4. Train the Model
    print(f"Training HistGradientBoostingRegressor on {len(X_train)} rows...")
    # !! CRITICAL: No cross-val, fitting on ALL data
    regression_pipeline.fit(X_train, y_train)
    
    train_time = time.time()
    print(f"...Model training complete. Time: {train_time - start_time:.2f}s")

    # 5. Prepare Test Data (X_test)
    # (Note: test set lacks 'amt' and 'is_fraud', so we just drop 'trans_num')
    X_test = processed_test_df.drop(columns=['trans_num'])

    # 6. Make Predictions
    print("Making predictions on the test set (test.csv)...")
    log_predictions = regression_pipeline.predict(X_test)

    # 7. !! CRITICAL: Invert the prediction!!
    # The model predicted log(amt), we must use expm1() to convert it back
    final_predictions = np.expm1(log_predictions)
    
    # (Safety check: amount cannot be negative)
    final_predictions[final_predictions < 0] = 0

    # 8. Create and Save Submission File
    print("Creating submission file...")
    submission_df = pd.DataFrame({
        'trans_num': test_ids,
        'amt': final_predictions
    })
    
    # !! IMPORTANT: Change z1234567 to your student ID !!
    submission_filename = 'z1234567_regression.csv' 
    submission_df.to_csv(submission_filename, index=False)

    end_time = time.time()
    print("\n--- Part II (Regression) SCRIPT COMPLETED SUCCESSFULLY! ---")
    print(f"Final file saved as: {submission_filename}")
    print(f"Total script runtime: {end_time - start_time:.2f} seconds.")
    print(f"(Must be < 120 seconds)")