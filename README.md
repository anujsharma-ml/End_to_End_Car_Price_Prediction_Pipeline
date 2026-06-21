CAR PRICE PREDICTION PIPELINE

Project Overview
This project implements an end-to-end machine learning pipeline to predict used car prices based on vehicle features. It focuses on clean code, proper data preprocessing, and advanced modeling.

Project Directory Structure
- data: Contains the raw dataset file (UserCarData.csv)
- Notebook: Contains the complete Jupyter Notebook with EDA and model training
- README.md: Project documentation file

Core Implementation Steps
1. Custom Transformer
Built a custom IQR-based outlier clipping class to handle extreme values automatically without causing any data leakage.

2. Target Transformation
Applied a logarithmic transformation (log1p) on the selling price to reduce skewness and improve model alignment.

3. Pipeline Processing
Structured the entire data preprocessing using ColumnTransformer for robust scaling and smart categorical encoding.

4. Ensemble Learning
Combined tuned models of Random Forest, SVR, and XGBoost using Voting and Stacking Regressors to get the best accuracy.

Tech Stack Used
- Programming Language: Python
- Data Manipulation: Pandas, NumPy
- Data Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-Learn, XGBoost