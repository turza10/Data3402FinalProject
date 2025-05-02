# Breast Cancer Diagnosis Prediction (WiDS 2024 Challenge)

**One Sentence Summary:** This repository contains a predictive modeling pipeline built to classify breast cancer diagnosis timing using health, demographic, and environmental features, as part of the WiDS Datathon 2024 challenge.

## Overview

The task, as defined by the WiDS 2024 Kaggle challenge, was to predict whether a breast cancer diagnosis occurred within 90 days based on a rich dataset containing patient characteristics, diagnosis codes, geographic and socioeconomic data, and environmental toxicity metrics. Our approach involved rigorous data preprocessing, feature engineering, exploratory data analysis, and the implementation of Logistic Regression and Random Forest models. Our best model achieved an ROC AUC of \~0.77 on validation data.

## Summary of Workdone

### Data

* **Type:** CSV files with structured tabular data

  * Input: patient-level demographic, diagnostic, environmental, and geographic features
  * Output: binary label (`DiagPeriodL90D`) indicating early (<90 days) or late diagnosis
* **Size:** \~18,000 rows in `training.csv`; a separate `test.csv` file for inference

### Preprocessing / Cleanup

* Dropped columns with over 80% missing values
* Imputed:

  * Numerical columns with training-set mean
  * Categorical columns with mode
* Applied Label Encoding to all categorical features
* Clipped outliers in key numeric features using IQR bounds

### Data Visualization

* Plotted missing value bar chart
* Visualized distributions of `patient_age`, `bmi`, `PM25`, `Ozone`, `income`, and `education` against the target
* Generated correlation matrix heatmaps

### Problem Formulation

* **Input:** 70+ features including demographics, pollution levels, and diagnosis codes
* **Output:** Binary classification of diagnosis period (<90 days or not)
* **Models Tried:**

  * Logistic Regression with feature scaling
  * Random Forest Classifier (baseline and randomized hyperparameter search)

### Training

* Performed train/validation split (80/20) using `train_test_split`
* Applied `StandardScaler` to Logistic Regression pipeline
* Used `RandomizedSearchCV` (3-fold CV) for Random Forest tuning to reduce computation time
* Used sklearn models trained in Google Colab (Python 3.10 environment)

### Performance Comparison

| Model                 | ROC AUC | Notes                                                           |
| --------------------- | ------- | --------------------------------------------------------------- |
| Logistic Regression   | \~0.76  | Strong recall, slight overfitting without regularization tuning |
| Random Forest (tuned) | \~0.77  | Best performance overall, robust to outliers                    |

* Plotted ROC curves for both models on validation set

### Conclusions

* Random Forest performed better than Logistic Regression without requiring feature scaling
* Feature engineering and handling missing data contributed significantly to model performance

### Future Work

* Add SHAP feature importance visualizations
* Try LightGBM and XGBoost models
* Experiment with PCA and feature reduction
* Submit predictions on Kaggle leaderboard

## How to Reproduce Results

1. Clone this repo and open the notebook in Colab or Jupyter
2. Install required packages (see Software Setup)
3. Run notebook top to bottom:

   * Preprocessing → EDA → Model Training → Evaluation
4. Place `training.csv` and `test.csv` in the root directory

### Overview of Files in Repository

* `Cancer_Diagnosis_Final_Project.ipynb`: full end-to-end pipeline
* `training.csv`: training data (not included; from Kaggle challenge)
* `test.csv`: test data (not included; from Kaggle challenge)

### Software Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Data

Download from: [WiDS Kaggle Challenge 2024](https://www.kaggle.com/competitions/widsdatathon2024/data)

### Training

* Run all preprocessing and model training cells
* Use `RandomizedSearchCV` block to fine-tune the Random Forest model

### Performance Evaluation

* View printed confusion matrix, classification report, and AUC
* Run ROC curve plots

## Citations

* [WiDS Datathon 2024](https://www.kaggle.com/competitions/widsdatathon2024)
* Scikit-learn documentation

