import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

st.title('Breast Cancer Classification Using XGBoost')
deskripsi = """
<p style='text-align: justify;'>
Dataset Breast Cancer Wisconsin adalah kumpulan data yang sering digunakan dalam penelitian klasifikasi dan deteksi kanker payudara. Dataset ini berisi fitur-fitur yang diekstraksi dari gambar digital hasil biopsi jarum halus (fine needle aspiration) dari massa payudara. 
Setiap sampel dalam dataset direpresentasikan oleh 30 atribut numerik yang menggambarkan karakteristik sel, seperti radius, tekstur, perimeter, luas, kehalusan, dan simetri inti sel. Dataset ini juga menyertakan label diagnosis yang diklasifikasikan sebagai benign (jinak) atau malignant (ganas).
</p>
"""

st.markdown(deskripsi, unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload file CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Pra-pemrosesan 
    df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    df.rename(columns={"diagnosis": "target"}, inplace=True)
    
    
    st.subheader("Data Sample")
    st.write(df.head())

    # Visualisasi distribusi
    st.subheader("Distribusi Target")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='target', palette="YlGnBu", ax=ax1)
    st.pyplot(fig1)

    # Pembagian data
    df["target"] = [1 if i.strip() == "M" else 0 for i in df.target]
    x = df.drop(['target'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # Standarisasi
    scaler = StandardScaler()
    X_Train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_Test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Model
    eval_set = [(X_Train, y_train), (X_Test, y_test)]
    xgb = XGBClassifier(objective='binary:logistic', learning_rate=0.5, max_depth=5, n_estimators=180, eval_metric=["logloss","error","auc"])
    xgb.fit(X_Train, y_train, eval_set=eval_set, verbose=False)

    y_pred = xgb.predict(X_Test)

    # Evaluasi
    st.subheader("Evaluasi Model")
    acc = accuracy_score(y_train, xgb.predict(X_Train))
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("**Akurasi pada data latih**:", acc)
    st.write("**Akurasi pada data uji**:", accuracy_score(y_test,y_pred))
    st.dataframe(pd.DataFrame(report).transpose())

    # Feature Importance
    st.subheader("Feature Importance")
    from xgboost import plot_importance
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    plot_importance(xgb, importance_type='gain', max_num_features=10, ax=ax2)
    st.pyplot(fig2)

    results = xgb.evals_result()
    df_score = pd.DataFrame({
        'iteration': range(len(results['validation_0']['logloss'])),
        'train_logloss': results['validation_0']['logloss'],
        'train_error': results['validation_0']['error'],
        'train_auc': results['validation_0']['auc'],
        'test_logloss': results['validation_1']['logloss'],
        'test_error': results['validation_1']['error'],
        'test_auc': results['validation_1']['auc'],
    })

    st.subheader("Hasil Evaluasi per Iterasi (Training vs Testing)")
    st.dataframe(df_score)  # interaktif

    st.subheader("Plot Logloss per Iterasi")
    fig, ax = plt.subplots()
    ax.plot(df_score['iteration'], df_score['train_logloss'], label='Train Logloss')
    ax.plot(df_score['iteration'], df_score['test_logloss'], label='Test Logloss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Logloss')
    ax.legend()
    st.pyplot(fig)

        #Kustomisasi Skema
    def get_scaler(name):
        if name == "StandardScaler":
            return StandardScaler()
        elif name == "MinMaxScaler":
            return MinMaxScaler()
        elif name == "RobustScaler":
            return RobustScaler()
        else:
            return None

    def build_pipeline(scaler_name, use_pca, n_components, max_depth, n_estimators, learning_rate, reg_alpha, reg_lambda):
        steps = []
        scaler = get_scaler(scaler_name)
        if scaler is not None:
            steps.append(('scaler', scaler))
        if use_pca:
            steps.append(('pca', PCA(n_components=n_components)))
        steps.append(('model', XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )))
        return Pipeline(steps)

    def evaluate_pipeline(pipeline, X_test, y_test):
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return acc, report

    def plot_metrics_comparison(results):
        # results = list of (idx, acc, report)
        metrics = ['precision', 'recall', 'f1-score']
        classes = ['0', '1']  # kelas negatif dan positif
    
        # Prepare dataframe untuk plot
        data = []
        for idx, acc, report in results:
            for cls in classes:
                row = {'Pipeline': f'Pipeline {idx}', 'Class': cls}
                for m in metrics:
                    row[m] = report[cls][m]
                data.append(row)
        df = pd.DataFrame(data)
    
        st.subheader("Perbandingan Akurasi")
        fig, ax = plt.subplots()
        sns.barplot(data=pd.DataFrame(results, columns=['idx','accuracy','report'])[['idx','accuracy']], x='idx', y='accuracy', ax=ax)
        ax.set_xticklabels([f'Pipeline {i}' for i in df['Pipeline'].unique()])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)
    
        st.subheader("Perbandingan Precision, Recall, F1-score per Kelas")
        for metric in metrics:
            fig, ax = plt.subplots()
            sns.barplot(data=df, x='Pipeline', y=metric, hue='Class', ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title(metric.capitalize())
            st.pyplot(fig)
    
    st.subheader("Kustomisasi 3 Pipeline XGBoost")

    def pipeline_input(prefix):
        scaler_name = st.selectbox("Scaler", ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"], key=f"scaler_{prefix}")
        
        use_pca = st.checkbox("Gunakan PCA?", key=f"pca_use_{prefix}")
        if use_pca:
            n_components = st.slider("Jumlah komponen PCA", 1, min(X_train.shape[1], 30), 10, key=f"pca_comp_{prefix}")
        else:
            n_components = None
    
        max_depth = st.slider("Max Depth", min_value=1, max_value=10, value=5, step=1, key=f"max_depth_{prefix}")
        n_estimators = st.slider("Jumlah Estimator", min_value=50, max_value=200, value=100, step=10, key=f"n_estimators_{prefix}")
        learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01, key=f"learning_rate_{prefix}")
        
        reg_alpha = st.number_input("reg_alpha (L1)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key=f"reg_alpha_{prefix}")
        reg_lambda = st.number_input("reg_lambda (L2)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key=f"reg_lambda_{prefix}")
        
        return scaler_name, use_pca, n_components, max_depth, n_estimators, learning_rate, reg_alpha, reg_lambda

    st.markdown("### Pipeline 1")
    p1 = pipeline_input("1")
    st.markdown("### Pipeline 2")
    p2 = pipeline_input("2")
    st.markdown("### Pipeline 3")
    p3 = pipeline_input("3")
    
    if st.button("Train Ketiga Pipeline"):
            pipelines = []
            pipelines.append(build_pipeline(*p1))
            pipelines.append(build_pipeline(*p2))
            pipelines.append(build_pipeline(*p3))
    
            results = []
            for i, pipe in enumerate(pipelines, 1):
                pipe.fit(X_train, y_train)
                acc, report = evaluate_pipeline(pipe, X_test, y_test)
                results.append((i, acc, report))
    
            for idx, acc, report in results:
                st.subheader(f"Hasil Evaluasi Pipeline {idx}")
                st.write(f"Akurasi: {acc:.4f}")
                st.text(classification_report(y_test, pipelines[idx-1].predict(X_test)))
    
            plot_metrics_comparison(results)
