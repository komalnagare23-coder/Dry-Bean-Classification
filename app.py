import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")
st.title("🫘 Dry Bean Classification")
st.write("Predict the type of dry bean using five machine-learning classifiers trained on the UCI Dry Bean dataset.")

MODEL_FILES = {
    "Logistic Regression": "Logistic_Regression.joblib",
    "Decision Tree": "Decision_Tree.joblib",
    "KNN": "KNN.joblib",
    "Naive Bayes": "Naive_Bayes.joblib",
    "SVM": "SVM.joblib",
}
FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2",
    "ShapeFactor3", "ShapeFactor4",
]

@st.cache_resource
def load_model(filename):
    # Load only trusted model artifacts bundled with this project.
    return joblib.load(Path(__file__).parent / "model" / filename)

st.subheader("Model comparison (held-out test set)")
metrics_path = Path(__file__).parent / "model_comparison.csv"
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    st.dataframe(metrics.style.format({c: "{:.4f}" for c in metrics.columns if c != "Model"}), use_container_width=True)
    best = metrics.loc[metrics["Accuracy"].idxmax()]
    st.success(f"Highest test accuracy: {best['Model']} ({best['Accuracy']:.2%})")
else:
    st.info("Add model_comparison.csv to show the evaluation table.")

st.subheader("Make predictions")
selected = st.selectbox("Choose a classification model", list(MODEL_FILES))
uploaded = st.file_uploader("Upload a CSV with the 16 numerical features", type=["csv"])
st.caption("Use the supplied test_data.csv as a sample. The file may contain multiple rows; a Class column, if present, is ignored.")

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        missing = [col for col in FEATURES if col not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        elif data.empty:
            st.warning("The uploaded CSV contains no data rows.")
        else:
            features = data[FEATURES].apply(pd.to_numeric, errors="coerce")
            if features.isna().any().any():
                st.error("Some feature values are missing or non-numeric. Please correct the CSV.")
            else:
                model = load_model(MODEL_FILES[selected])
                predictions = model.predict(features)
                output = data.copy()
                output["Predicted_Class"] = predictions
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)
                    output["Prediction_Confidence"] = probs.max(axis=1).round(4)
                st.success(f"Predicted {len(output)} row(s) using {selected}.")
                st.dataframe(output, use_container_width=True)
                st.download_button("Download predictions CSV", output.to_csv(index=False).encode("utf-8"), "dry_bean_predictions.csv", "text/csv")
                if "Class" in data.columns: 
                                from sklearn.metrics import (
                                accuracy_score, roc_auc_score, precision_score,
                                recall_score, f1_score, matthews_corrcoef
                                )

                                y_true = data["Class"]
                                y_pred = predictions

                                accuracy = accuracy_score(y_true, y_pred)
                                precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                                recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                                f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                                mcc = matthews_corrcoef(y_true, y_pred)
                    
                                from sklearn.preprocessing import label_binarize
                                classes = model.classes_
                                y_binary = label_binarize(y_true, classes=classes)
                                probabilities = model.predict_proba(features)
                                auc = roc_auc_score(y_binary, probabilities, average="weighted", multi_class="ovr")
                    
                                st.subheader("Evaluation Metrics")
                                st.dataframe(pd.DataFrame({
                                "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                                "Score": [accuracy, auc, precision, recall, f1, mcc]
                                }))
                                st.subheader("Classification Report")
                                st.text(classification_report(data["Class"], predictions, zero_division=0))
                                st.subheader("Confusion Matrix")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                from sklearn.metrics import ConfusionMatrixDisplay
                                ConfusionMatrixDisplay.from_predictions(data["Class"], predictions, ax=ax, xticks_rotation=45)
                                st.pyplot(fig)
                                plt.close(fig)
                
                
    except Exception as exc:
        st.error(f"Unable to process this CSV: {exc}")

st.divider()
st.caption("BITS Pilani WILP · Machine Learning Assignment 2 · Student ID: 2026AJ05588")
