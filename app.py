import streamlit as st
import pandas as pd
import numpy as np
import pickle, io, os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide", page_title="Model Comparison App")

st.title("Model Viewer — Load pre-trained .pkl models and compare performance")

def load_data(file):
    return pd.read_csv(file)

def preprocess_data(X):
    X_processed = X.copy()
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object' or X_processed[col].dtype.name == 'category':
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    return X_processed

# Helper to load all model pickles in CWD
def load_all_models():
    model_files = [p for p in Path('.').glob('*.pkl') if p.name != 'model_metadata.pkl']
    models = {}
    for p in model_files:
        try:
            with open(p, 'rb') as f:
                models[p.stem] = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed loading {p.name}: {e}")
    return models

uploaded = st.file_uploader("Upload your CSV", type=['csv'])
if uploaded is None:
    st.info("Upload CSV (same structure used during training).")
else:
    df = load_data(uploaded)
    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    # Identify features/target (assume last column target)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split so evaluation matches training split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load models
    models = load_all_models()
    if not models:
        st.error("No model .pkl files found in the app folder. Run the training notebook to create them.")
    else:
        st.sidebar.subheader("Model selection")
        model_names = list(models.keys())
        selected = st.sidebar.selectbox("Choose a model to view", model_names)

        st.sidebar.write("Or click below to compare all models:")
        compare_btn = st.sidebar.button("Compare all models")

        if selected:
            model = models[selected]
            # Preprocess test data same as training
            X_test_proc = preprocess_data(X_test)
            y_test_proc = y_test.copy()
            if y_test_proc.dtype == 'object' or y_test_proc.dtype.name == 'category':
                le_y = LabelEncoder()
                y_test_proc = le_y.fit_transform(y_test_proc.astype(str))

            y_pred = model.predict(X_test_proc)

            acc = accuracy_score(y_test_proc, y_pred)
            prec = precision_score(y_test_proc, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test_proc, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_proc, y_pred, average='weighted', zero_division=0)
            st.subheader(f"Results for {selected}")
            st.write(f"Accuracy: **{acc:.4f}**")
            st.write(f"Precision (weighted): **{prec:.4f}**")
            st.write(f"Recall (weighted): **{rec:.4f}**")
            st.write(f"F1 Score (weighted): **{f1:.4f}**")

            
            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_test_proc, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            st.pyplot(fig)

        if compare_btn:
            # Evaluate all models and show comparison bar chart
            results = []
            for name, model in models.items():
                X_test_proc = preprocess_data(X_test)
                y_test_proc = y_test.copy()
                if y_test_proc.dtype == 'object' or y_test_proc.dtype.name == 'category':
                    le_y = LabelEncoder()
                    y_test_proc = le_y.fit_transform(y_test_proc.astype(str))
                y_pred = model.predict(X_test_proc)
                results.append({
                    'Model': name,
                    'Accuracy': accuracy_score(y_test_proc, y_pred),
                    'Precision': precision_score(y_test_proc, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test_proc, y_pred, average='weighted', zero_division=0),
                    'F1 Score': f1_score(y_test_proc, y_pred, average='weighted', zero_division=0)
                })
            res_df = pd.DataFrame(results).set_index('Model')
            st.subheader('Comparison of models (on uploaded test split)')
            st.dataframe(res_df)
            st.subheader('Metric comparison chart')
            # Bar chart for each metric
            fig2, ax2 = plt.subplots(figsize=(10,5))
            res_df.plot(kind='bar', ax=ax2)
            ax2.set_ylabel('Score')
            ax2.set_ylim(0,1.05)
            st.pyplot(fig2)







# st.subheader("Classification report")
            # report = classification_report(y_test_proc, y_pred, output_dict=True, zero_division=0)
            # rep_df = pd.DataFrame(report).transpose()
            # st.dataframe(rep_df)
