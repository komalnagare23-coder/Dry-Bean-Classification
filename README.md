# Dry Bean Classification — Machine Learning Assignment 2

**Student ID:** 2026AJ05588  
**Course:** BITS Pilani WILP — Machine Learning

## Problem statement
Predict the variety of dry bean from 16 measured numerical characteristics using supervised classification. Compare five models on the same stratified train/test split and provide an interactive web app for predictions.

## Dataset
**UCI Dry Bean Dataset:** 13,611 observations, 16 numeric input features, and seven bean classes. The dataset was read from `Dry_Bean_Dataset.xlsx` in the BITS Virtual Lab. The split used 80% training (10,888 rows) and 20% testing (2,723 rows), `random_state=42`, stratified by class.

## Algorithms
Logistic Regression (standardized), Decision Tree, K-Nearest Neighbors (standardized), Gaussian Naive Bayes, and Support Vector Machine (standardized; probability estimates enabled).

## Results
Metrics below are from the BITS lab execution screenshot; precision, recall, F1 and multiclass one-vs-rest AUC are weighted averages.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9934 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.8920 | 0.9334 | 0.8917 | 0.8920 | 0.8916 | 0.8696 |
| KNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| SVM | **0.9221** | **0.9937** | **0.9224** | **0.9221** | **0.9222** | **0.9059** |

## Observations and best model


| Model | Performance observation |
|---|---|
| Logistic Regression | Achieved 92.14% accuracy and 0.9934 AUC, showing strong classification performance. |
| Decision Tree | Achieved 89.20% accuracy, performing below Logistic Regression, KNN, and SVM. |
| KNN | Achieved 91.66% accuracy, with strong F1 and MCC scores. |
| Naive Bayes | Achieved 76.39% accuracy, the lowest among the five models. |
| SVM | Achieved the highest accuracy (92.21%), AUC (0.9937), F1 (0.9222), and MCC (0.9059). |
| Overall winner | **SVM**, based on the highest test accuracy and overall evaluation metrics. |

## Streamlit application
The Streamlit application allows users to select any of the five trained classifiers, upload CSV data, generate predictions, view all six evaluation metrics (Accuracy, AUC, Precision, Recall, F1, and MCC), inspect the classification report and confusion matrix, and download predictions. The application also displays a comparison of all five models.


### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, `model_comparison.csv`, and the `model/` folder.
2. Include `test_data.csv` exported from the BITS lab, containing the 16 feature columns.
3. In Streamlit Community Cloud, create a new app from that repository and select `app.py` as the main file.
4. Verify CSV predictions using `test_data.csv`.

## Links
- **GitHub repository:https://github.com/komalnagare23-coder/Dry-Bean-Classification
- **Live Streamlit app:https://dry-bean-classification-dfqieqj2heawhvlv9cczlp.streamlit.app

## Evidence
Include a screenshot showing successful execution in the BITS Virtual Lab in the final assignment PDF.

## Model artifacts
The `model/` folder contains five `.joblib` models trained in the BITS Virtual Lab. Load only trusted model files. Models were created with scikit-learn 1.5.2; the deployment requirements pin that version.
