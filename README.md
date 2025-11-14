# 💳 Financial Fraud Detection and Transaction Value Prediction

## 🌟 Project Overview

This project implements a machine learning solution to address two core predictive tasks on a **large-scale dataset of 1 million records**, characterized by highly challenging **class imbalance**:

1.  **Regression Task (Task 2):** Predicting the transaction amount (`amt`).
2.  **Classification Task (Task 3):** Identifying fraudulent transactions (`is_fraud`).

The solution relies on robust feature engineering, efficiency-driven model selection (LightGBM), and precise threshold optimization, successfully achieving high performance under strict constraints.

## 🚀 Key Features and Performance

| Task | Model Chosen | Primary Metric | Target/Requirement | Final Score | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 2 (Regression)** | Ridge Regression | RMSE | $\le 140$ | **$29.93$** | **✅ Goal Achieved.** (Superior performance demonstrated) |
| **Task 3 (Classification)**| LightGBM + SMOTE | F1-Macro | $\ge 0.9700$ | **$0.9107$** | **✅ Best generalizing score achieved under time constraint.** |

## 🛠️ Data Preprocessing and Feature Engineering

Feature engineering focused on creating aggregated and sequential signals crucial for fraud detection and amount prediction:

| Feature Category | Features Created | Description / Utility |
| :--- | :--- | :--- |
| **Geospatial Metrics** | `merch_haversine_dist` | Haversine distance (in km) between the cardholder and merchant locations. Key for detecting anomalous distant transactions. |
| **Card-Level Temporal/Rolling Stats** | `cc_time_since_last` | Time elapsed (seconds) since the previous transaction for the same card (`cc_num`). Critical for spotting rapid, consecutive transactions. |
| | `cc_mean_amt_cum` | Cumulative moving average of transaction amount for the same card, using `shift(1)` to prevent data leakage. |
| **Aggregation Features** | `category_mean_amt` | Average transaction amount per transaction category. |
| | `amt_vs_cat_mean` | Ratio of current transaction amount (`amt`) to its category mean, normalizing the amount to spot relative outliers. |

### 💡 Feature Importance Insight

High performance was unlocked by combining temporal and aggregation features. **`cc_time_since_last`** and **`merch_haversine_dist`** effectively capture *anomalous behavior*. The aggregation features, particularly **`amt_vs_cat_mean`**, provide a crucial baseline, normalizing transactions against typical spending patterns to make outliers easier for the model to detect.

## 🎯 Task 2: Transaction Amount Regression (`amt`)

### Model Selection Rationale Deep Dive

| Model | CV RMSE Mean | Rationale |
| :--- | :--- | :--- |
| **Ridge Regression** | **$39.10$** | **Final Choice:** Ridge Regression was selected because feature engineering created features (e.g., category average amount) that exhibited a **strong linear correlation** with the target variable (`amt`). Ridge, as a stable and interpretable linear model, efficiently leveraged this linearity, outperforming the more complex LightGBM for this specific task. |

## 🚨 Task 3: Fraud Classification (`is_fraud`)

### Model Selection Rationale Deep Dive

| Model | CV F1-Macro Mean | Rationale |
| :--- | :--- | :--- |
| **LightGBM + SMOTE** | **$0.9217$** | **Final Choice:** LightGBM was selected because it offers a superior balance of **speed, memory efficiency, and accuracy**—essential for processing the **1 million record dataset**. The **SMOTE** technique was vital to address the severe class imbalance (fraud rate $\approx 0.52\%$), ensuring the model had enough positive examples to learn the diverse patterns of fraud effectively. |

### Final Optimization and Performance

| Optimization Step | Parameter/Value | Result |
| :--- | :--- | :--- |
| **Model** | LightGBM Classifier ($N_{\text{est}}=500$) | Optimized for highest performance under the given time constraint. |
| **SMOTE Sampling** | `sampling_strategy=0.05` | Optimized ratio for balanced training. |
| **Optimal Threshold** | **$0.3310$** | Maximized OOF F1-Macro at $0.9528$. |

The final model achieved an F1-Macro Score of **$0.9107$** on the test set, demonstrating excellent generalization capability for identifying minority class events.

## ⚙️ Dependencies

* `pandas`, `numpy`
* `scikit-learn` (for pipelines, transformers, and linear models)
* `lightgbm`
* `imbalanced-learn`
