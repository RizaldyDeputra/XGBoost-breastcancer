import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

st.title('Breast Cancer Classification Using XGBoost')

# Upload File Dataset
uploaded_file = st.file_uploader("Upload file CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Pra-pemrosesan 
    df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    df.rename(columns={"diagnosis": "target"}, inplace=True)
    df["target"] = [1 if i.strip() == "M" else 0 for i in df.target]
    
    st.subheader("Data Sample")
    st.write(df.head())

    # Visualisasi distribusi
    st.subheader("Distribusi Target")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='target', palette="YlGnBu", ax=ax1)
    st.pyplot(fig1)

    # Pembagian data
    x = df.drop(['target'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # Standarisasi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    xgb = XGBClassifier(objective='binary:logistic', learning_rate=0.5, max_depth=5, n_estimators=180, eval_metric=["logloss","error","auc"])
    xgb.fit(X_train_scaled, y_train)

    y_pred = xgb.predict(X_test_scaled)

    # Evaluasi
    st.subheader("Evaluasi Model")
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(f"**Akurasi**: {acc:.4f}")
    st.dataframe(pd.DataFrame(report).transpose())

    # Feature Importance
    st.subheader("Feature Importance")
    from xgboost import plot_importance
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    plot_importance(xgb, importance_type='gain', max_num_features=10, ax=ax2)
    st.pyplot(fig2)
