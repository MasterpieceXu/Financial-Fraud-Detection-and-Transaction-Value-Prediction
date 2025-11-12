## ### Final Project Submission Script: Full Integration (Part II Ridge + Part III LGBM) ###

import pandas as pd
import numpy as np
from datetime import datetime
import time
import  sys
# Scikit-learn & Imblearn Libraries
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score # For optional test set evaluation

# ======================================================================
## Part 0: Parameter Setup and Preprocessing Functions
# ======================================================================

REFERENCE_DATE = pd.to_datetime('2021-01-01')
SAFE_N_JOBS = 2 # For parallel processing

# --- 1. Data Preprocessing Functions (from your code) ---
def preprocess_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning, type conversion, and time/age feature creation."""
    df = df.copy()
    df['trans_num'] = df['trans_num'].astype(str)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_day'] = df['trans_date_trans_time'].dt.dayofweek
    df['trans_month'] = df['trans_date_trans_time'].dt.month
    df['trans_year'] = df['trans_date_trans_time'].dt.year
    df['age'] = (REFERENCE_DATE - df['dob']).dt.days // 365
    cols_to_drop = ['first', 'last', 'street', 'dob', 'unix_time']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates Haversine distance between two sets of coordinates."""
    R=6317
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Creates advanced distance and rolling statistics features."""
    df = df.copy()
    df['merch_haversine_dist'] = haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    df = df.drop(columns=['lat', 'long', 'merch_lat', 'merch_long'], errors='ignore')
    df['unix_time_sec'] = df['trans_date_trans_time'].astype(np.int64) // 10**9
    df['cc_time_since_last'] = df.groupby('cc_num')['unix_time_sec'].diff().fillna(999999) 
    df['cc_count_cum'] = df.groupby('cc_num').cumcount()
    df['cc_mean_amt_cum'] = df.groupby('cc_num')['amt'].transform(
        lambda x: x.shift(1).expanding().mean().fillna(0)
    )
    first_time = df.groupby('cc_num')['unix_time_sec'].transform('min')
    df['cc_time_diff_total'] = df['unix_time_sec'] - first_time + 1 
    df['cc_freq'] = df['cc_count_cum'] / df['cc_time_diff_total']
    df['category_mean_amt'] = df.groupby('category')['amt'].transform('mean')
    df['amt_vs_cat_mean'] = df['amt'] / df['category_mean_amt']
    df['amt_per_pop'] = df['amt'] / (df['city_pop'] + 1)
    df = df.drop(columns=['city', 'state', 'zip', 'unix_time_sec', 'cc_time_diff_total', 'cc_num', 'merchant'], errors='ignore')
    return df

# --- 2. Final Parameter Settings ---
# Part II (Ridge) Final Parameters
FINAL_REG_ALPHA = 10.0 # Optimal alpha determined from tuning

# Part III (LGBM) Final Parameters (Best generalizing configuration)
FINAL_CLS_THRESHOLD = 0.3310      # Optimal threshold (from OOF with SMOTE=0.05)
FINAL_CLS_ESTIMATORS = 500        # Maximum safe tree count
MAX_DEPTH_FINAL = 15
NUM_LEAVES_FINAL = 70
FINAL_SMOTE_SAMPLING = 0.05       # Best generalizing SMOTE ratio
LEARNING_RATE_FINAL = 0.05


# ======================================================================
## Part I: Data Loading, Feature Engineering, and Split
# ======================================================================

if len(sys.argv) != 3:
    print("Usage: python z{studentID}.py <train_csv_path> <test_csv_path>")
    sys.exit(1)

TRAIN_FILE_PATH = sys.argv[1]
TEST_FILE_PATH = sys.argv[2]



print("--- 1/5: Data Loading and Feature Engineering ---")
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Error: Ensure train.csv and test.csv are in the current working directory.")
    exit()

df_full = feature_engineering(preprocess_basic(train_df))
df_test_proc = feature_engineering(preprocess_basic(test_df))

# Define Features and Targets
TARGET_REG = 'amt'
TARGET_CLS = 'is_fraud'
FEATURES_COMMON = [col for col in df_full.columns if col not in [TARGET_REG, TARGET_CLS, 'trans_num', 'trans_date_trans_time']]

# Training Data Split
X_train = df_full[FEATURES_COMMON]
y_train_reg = df_full[TARGET_REG]    # Part II Regression Target
y_train_cls = df_full[TARGET_CLS]    # Part III Classification Target

# Testing Data Split
X_test = df_test_proc[FEATURES_COMMON]
transaction_ids = df_test_proc['trans_num'] # Transaction IDs for final submission

# Define Preprocessor
numeric_features = [
    'city_pop', 'age', 'merch_haversine_dist', 'cc_time_since_last', 
    'cc_mean_amt_cum', 'cc_count_cum', 'cc_freq', 'category_mean_amt', 
    'amt_vs_cat_mean', 'amt_per_pop'
] 
categorical_features = [
    'category', 'gender', 'job', 'trans_hour', 'trans_day', 
    'trans_month', 'trans_year' 
] 

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough' 
)
# Alias for consistency with Part III pipeline definition
preprocessor_reg = preprocessor 


# ======================================================================
## Part II: Ridge Regression Training and Prediction (Generates fraud_likelihood)
# ======================================================================

print("\n--- 2/5: Part II Ridge Regression Training and Prediction (fraud_likelihood) ---")

# 1. Create Ridge Model (using optimal alpha)
final_ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_reg),
    ('regressor', Ridge(alpha=FINAL_REG_ALPHA, random_state=42))
])

# 2. Train Model
start_time_reg = time.time()
final_ridge_pipeline.fit(X_train, y_train_reg)
training_time_reg = time.time() - start_time_reg
print(f"✅ Ridge Model Training Time: {training_time_reg:.2f} seconds")

# 3. Predict on test.csv
test_predictions_reg = final_ridge_pipeline.predict(X_test)
# Correct negative values
test_predictions_reg[test_predictions_reg < 0] = 0 
# Result variable: test_predictions_reg


# ======================================================================
## Part III: LightGBM Classification Training and Prediction (Generates is_fraud)
# ======================================================================

print("\n--- 3/5: Part III LightGBM Classification Training and Prediction (is_fraud) ---")

# 1. Create LightGBM Model (using final parameters)
final_lgbm_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor_reg),
    ('smote', SMOTE(sampling_strategy=FINAL_SMOTE_SAMPLING, random_state=42)), 
    ('classifier', LGBMClassifier(
        n_estimators=FINAL_CLS_ESTIMATORS, 
        max_depth=MAX_DEPTH_FINAL,        
        num_leaves=NUM_LEAVES_FINAL,
        learning_rate=LEARNING_RATE_FINAL,
        random_state=42, n_jobs=1, verbose=-1))
])

# 2. Train Model
start_time_cls = time.time()
final_lgbm_pipeline.fit(X_train, y_train_cls)
training_time_cls = time.time() - start_time_cls
print(f"✅ LightGBM Model Training Time: {training_time_cls:.2f} seconds")

# 3. Predict probabilities on test.csv
test_proba = final_lgbm_pipeline.predict_proba(X_test)[:, 1]

# 4. Apply the optimized threshold for classification
test_predictions_cls = (test_proba >= FINAL_CLS_THRESHOLD).astype(int)
# Result variable: test_predictions_cls


# ======================================================================
## Part IV: Submission File Generation (Generating the two required files)
# ======================================================================

STUDENT_ID = "Your_file_name"

# --- 4/6: Generate Part II Regression File (z{studentID}_regression.csv) ---
OUTPUT_FILENAME_REG = f"z{STUDENT_ID}_regression.csv"
print(f"\n--- 4/5: Generating Part II Regression File: {OUTPUT_FILENAME_REG} ---")

submission_reg_df = pd.DataFrame({
    'trans_num': transaction_ids,
    'amt': test_predictions_reg,  
})
# Ensure column order: trans_num, amt
submission_reg_df = submission_reg_df[['trans_num', 'amt']]
submission_reg_df.to_csv(OUTPUT_FILENAME_REG, index=False)
print(f"🎉 Part II Regression submission file created successfully.")


# --- 5/6: Generate Part III Classification File (z{studentID}_classification.csv) ---
OUTPUT_FILENAME_CLS = f"z{STUDENT_ID}_classification.csv"
print(f"\n--- 5/5: Generating Part III Classification File: {OUTPUT_FILENAME_CLS} ---")

# Assuming classification output format requires trans_num and is_fraud
submission_cls_df = pd.DataFrame({
    'trans_num': transaction_ids,
    'is_fraud': test_predictions_cls          
})
submission_cls_df.to_csv(OUTPUT_FILENAME_CLS, index=False)
print(f"🎉 Part III Classification submission file created successfully.")



print(f"Total Estimated Training Time: {training_time_reg + training_time_cls:.2f} seconds (SAFE!)")

