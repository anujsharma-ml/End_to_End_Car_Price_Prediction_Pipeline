# AutoValue — Intelligent Car Price Predictor

[![Project Link](https://img.shields.io/badge/Project-Live_Demo-blue)](https://endtoendcarpricepredictionpipeline-rb8enhmfsdb352bflp3n7r.streamlit.app/)

AutoValue is an AI-powered machine learning application designed to estimate the market value of used cars. By leveraging a robust machine learning pipeline trained on comprehensive vehicle datasets, this tool provides accurate price predictions based on key vehicle parameters.

## 🚀 Project Overview

The project follows a complete machine learning lifecycle, starting from data ingestion and cleaning, through exploratory data analysis (EDA), to advanced feature engineering and model deployment.

- **Objective**: Predict the `selling_price` of used cars based on features like brand, fuel type, engine capacity, mileage, and car age.
- **Data Source**: Trained on `UserCarData.csv`, capturing vehicle trends from 1994 to 2020.
- **Deployment**: Built using **Streamlit** for a responsive, interactive, and modern user interface.

## ⚙️ Model Pipeline

The predictive engine utilizes a specialized pipeline to ensure high performance:

1.  **Preprocessing**:
    - **Categorical Handling**: One-hot encoding for car attributes (brand, transmission, fuel, etc.).
    - **Numerical Scaling**: Robust scaling and mean imputation to handle variations and missing data.
    - **Outlier Mitigation**: Custom `Iqrclipper` transformer to manage extreme values in vehicle data.
2.  **Transformation**:
    - `TransformedTargetRegressor` is employed to log-transform the target variable (`selling_price`), reducing skewness and improving model stability.
3.  **Model Selection**:
    - The final model is an **XGBoost Regressor**, selected for its superior performance (R² score of ~0.967) compared to Random Forest and SVR.
    - Saved and loaded via `joblib` for efficient inference.

## 🛠 Tech Stack

- **Language**: Python 3.12+
- **Machine Learning**: `scikit-learn` 1.6.1, `xgboost` 3.3.0
- **Web Interface**: Streamlit
- **Data Handling**: `pandas`, `numpy`
- **Model Persistence**: `joblib`

## 📊 Performance Metrics

| Model | R2 Score (Test Set) |
| :--- | :--- |
| **XGBoost Regressor** | **0.9674** |
| Random Forest | 0.9529 |
| Stacking Regressor | 0.9632 |
| Voting Regressor | 0.9546 |
| SVR | 0.9067 |

## 📦 How to Run

1.  **Clone the repository**:
    ```bash
    git clone [your-repo-url]
    cd [repo-directory]
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch the app**:
    ```bash
    streamlit run app.py
    ```

---
*Developed with focus on robust ML pipelines and interactive design.*
