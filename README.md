Predicting Calorie Burn Levels Using Machine Learning

Project Overview
This project investigates whether calorie burn levels (Low, Medium, High) can be predicted using biometric and workout-related features from fitness tracking data.
Dataset: Synthetic Fitness Tracker Dataset (Kaggle, 1,800 records, 15 features).
Key feature engineering: Heart_Rate_Range = Max_BPM − Resting_BPM.
Models: Logistic Regression, Random Forest (baseline + balanced), Support Vector Machines (linear and RBF), and Multilayer Perceptron (MLP).
Extensions: Regression trials (Linear Regression, Random Forest Regressor, SVR) and alternative imputation strategies (median/mode, drop-row, KNN).

Best result: Balanced Random Forest ~58% accuracy, outperforming other classifiers.

Requirements
Python 3.9+
Libraries:
pandas
numpy
scikit-learn
matplotlib
seaborn (optional for visualisation)

Results Summary
Logistic Regression → ~55% accuracy (biased toward Medium).
Random Forest → ~56% accuracy (baseline).
Balanced Random Forest → 58% accuracy (best performer).
SVM (Linear, RBF) → 31–36% accuracy, more balanced but weaker overall.
MLP Neural Network → ~55% accuracy, collapsed into majority class.
Regression methods → R² ≈ 0 (unsuitable for calorie prediction).

Future Work
Apply boosting models (XGBoost, LightGBM).
Address class imbalance with SMOTE or ensemble balancing.
Validate on real-world wearable datasets.
Engineer richer HR features (HRV measures).
