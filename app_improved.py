"""
Student Performance Predictor & Visualizer
A comprehensive Streamlit application for predicting and analyzing student performance
with AutoML, clustering, and model interpretability features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, Dict, Any, List, Optional

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, r2_score, confusion_matrix, 
    classification_report, mean_squared_error
)
from sklearn.cluster import KMeans

# ============== CONSTANTS ==============
APP_TITLE = "🎓 Student Performance Predictor & Visualizer"
PAGE_TITLE = "Student Performance Predictor"
TEST_SIZE = 0.2
RANDOM_STATE = 42
RISK_THRESHOLDS = {"high": 50, "medium": 75}
MIN_CLUSTERS = 2
MAX_CLUSTERS = 5
DEFAULT_CLUSTERS = 3
SAMPLE_SIZE = 100
N_ESTIMATORS = 100

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CUSTOM UI ==============
st.markdown("""
<style>
.main {background-color: #0E1117;}
h1 {text-align: center; color: #00D9FF; font-size: 2.5em;}
.metric-card {
    background: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
.success-card {background: #0d3d2c; padding: 10px; border-radius: 8px; color: #00FF41;}
.error-card {background: #3d0d0d; padding: 10px; border-radius: 8px; color: #FF0000;}
.tab-header {color: #00D9FF; font-weight: bold; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# ============== LOGIN SYSTEM ==============
USERS = {
    "admin": {"password": "admin123", "role": "Admin"},
    "teacher": {"password": "teach123", "role": "Teacher"}
}

def init_session_state() -> None:
    """Initialize session state variables."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.role = None
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

def login_page() -> None:
    """Display and handle login page."""
    st.markdown("<h2 style='text-align:center;'>🔐 Login Portal</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.write("Demo Credentials:")
        st.info("👤 Username: admin or teacher\n🔑 Passwords: admin123 or teach123")
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True):
            if username in USERS and USERS[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.role = USERS[username]["role"]
                st.success("✅ Login Successful! Redirecting...")
                st.rerun()
            else:
                st.error("❌ Invalid Credentials. Please try again.")

init_session_state()

if not st.session_state.logged_in:
    login_page()
    st.stop()

# ============== HEADER ==============
st.markdown(f"<h1>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.caption(f"👤 Logged in as: **{st.session_state.role}** | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

# ============== UTILITY FUNCTIONS ==============

def classify_risk(prediction: Any, is_classification: bool) -> str:
    """
    Classify prediction as low, medium, or high risk.
    
    Args:
        prediction: Model prediction value
        is_classification: Whether this is a classification task
        
    Returns:
        Risk level string with emoji
    """
    if is_classification:
        pred_str = str(prediction).lower()
        return "High Risk ❌" if pred_str in ["fail", "low", "poor"] else "Low Risk ✅"
    else:
        if prediction >= RISK_THRESHOLDS["medium"]:
            return "Low Risk ✅"
        elif prediction >= RISK_THRESHOLDS["high"]:
            return "Medium Risk ⚠️"
        else:
            return "High Risk ❌"

@st.cache_data
def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    is_classification: bool,
    _preprocessor: ColumnTransformer
) -> Tuple[Optional[Pipeline], str, float]:
    """
    Train multiple models and return the best one using AutoML.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        is_classification: Whether this is a classification task
        _preprocessor: Sklearn preprocessor pipeline (prefixed with _ to exclude from caching)
        
    Returns:
        Tuple of (best pipeline, model name, best score)
    """
    models: Dict[str, Any] = {}

    if is_classification:
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=N_ESTIMATORS, 
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        }
    else:
        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "LinearRegression": LinearRegression()
        }

    best_model, best_score, best_name = None, -1, ""

    for name, model in models.items():
        try:
            pipe = Pipeline([
                ('preprocessor', _preprocessor),
                ('model', model)
            ])

            pipe.fit(X_train, y_train)
            predictions = pipe.predict(X_test)

            score = (
                accuracy_score(y_test, predictions) 
                if is_classification 
                else r2_score(y_test, predictions)
            )

            if score > best_score:
                best_score = score
                best_model = pipe
                best_name = name
        except Exception as e:
            st.warning(f"⚠️ Error training {name}: {str(e)}")
            continue

    return best_model, best_name, best_score

def setup_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Create and return a preprocessing pipeline.
    
    Args:
        numeric_features: List of numeric feature column names
        categorical_features: List of categorical feature column names
        
    Returns:
        Configured ColumnTransformer for preprocessing
    """
    return ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

def create_shap_explanation(pipeline: Pipeline, X_sample: pd.DataFrame) -> None:
    """
    Create and display SHAP explanation plot.
    
    Args:
        pipeline: Trained sklearn pipeline
        X_sample: Sample data for explanation
    """
    try:
        X_transformed = pipeline.named_steps['preprocessor'].transform(X_sample)
        explainer = shap.TreeExplainer(pipeline.named_steps['model'])
        shap_values = explainer.shap_values(X_transformed)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_transformed, show=False)
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ SHAP explanation not available: {str(e)}")

# ============== SIDEBAR ==============
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("📤 Upload CSV File", type="csv")

# ============== MAIN APPLICATION ==============
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate data
        if df.empty:
            st.error("❌ Uploaded file is empty")
            st.stop()
        
        st.sidebar.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")

        # ============== COLUMN SETUP ==============
        st.sidebar.subheader("📋 Column Configuration")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            name_col = st.selectbox("Student Name Column", df.columns)
        with col2:
            id_col = st.selectbox("ID Column", df.columns)
        
        target_col = st.selectbox("Target Variable", df.columns)

        feature_cols = [c for c in df.columns if c not in [name_col, id_col, target_col]]

        numeric_features = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
        categorical_features = df[feature_cols].select_dtypes(include="object").columns.tolist()

        X = df[feature_cols]
        y = df[target_col]

        # Determine task type
        is_classification = y.dtype == 'object' or y.nunique() < 10

        # ============== DATA PREPROCESSING ==============
        preprocessor = setup_preprocessor(numeric_features, categorical_features)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE
        )

        # ============== MODEL TRAINING ==============
        use_automl = st.sidebar.checkbox("🤖 Enable AutoML", value=True)

        with st.spinner("⏳ Training model..."):
            if use_automl:
                pipeline, best_model_name, best_score = train_models(
                    X_train, X_test, y_train, y_test, is_classification, preprocessor
                )
                st.sidebar.success(f"✅ Best Model: **{best_model_name}**")
                st.session_state.pipeline = pipeline
            else:
                model = (
                    RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
                    if is_classification
                    else RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
                )
                pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
                pipeline.fit(X_train, y_train)
                best_model_name = "RandomForest"
                st.session_state.pipeline = pipeline

        y_pred = pipeline.predict(X_test)

        # ============== METRICS DISPLAY ==============
        st.subheader("📊 Model Performance")
        
        metric_cols = st.columns(3 if is_classification else 4)

        if is_classification:
            acc = accuracy_score(y_test, y_pred)
            metric_cols[0].metric("Accuracy", f"{acc:.2%}")
        else:
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = np.mean(np.abs(y_test - y_pred))
            
            metric_cols[0].metric("R² Score", f"{r2:.4f}")
            metric_cols[1].metric("RMSE", f"{rmse:.2f}")
            metric_cols[2].metric("MAE", f"{mae:.2f}")

        # ============== TABS ==============
        tabs = st.tabs([
            "👤 Student Prediction",
            "📊 Analytics",
            "📈 Model Details",
            "🧠 Model Explanation",
            "🧩 Clustering",
            "📂 Data View"
        ])

        # -------- TAB 1: STUDENT PREDICTION --------
        with tabs[0]:
            st.markdown("<h3 class='tab-header'>Student Performance Prediction</h3>", unsafe_allow_html=True)
            
            student_name = st.selectbox("Select Student", df[name_col].astype(str).unique())
            student_data = df[df[name_col].astype(str) == student_name].iloc[0]

            col1, col2, col3 = st.columns(3)
            
            col1.metric("📛 Name", student_name)
            col2.metric("🔢 ID", student_data[id_col])

            # Make prediction
            student_features = df[df[name_col].astype(str) == student_name][feature_cols]
            prediction = pipeline.predict(student_features)[0]
            
            col3.metric("🎯 Prediction", f"{round(prediction, 2)}" if not is_classification else prediction)

            # Risk assessment
            risk = classify_risk(prediction, is_classification)
            if "High" in risk:
                st.error(f"Risk Level: {risk}")
            elif "Medium" in risk:
                st.warning(f"Risk Level: {risk}")
            else:
                st.success(f"Risk Level: {risk}")

            # Show student profile
            with st.expander("📋 Student Profile"):
                st.dataframe(student_data.to_frame(), use_container_width=True)

        # -------- TAB 2: ANALYTICS --------
        with tabs[1]:
            st.markdown("<h3 class='tab-header'>Correlation Analysis</h3>", unsafe_allow_html=True)
            
            if numeric_features:
                corr_matrix = df[numeric_features].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu",
                    title="Feature Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No numeric features available for correlation analysis")

        # -------- TAB 3: MODEL DETAILS --------
        with tabs[2]:
            st.markdown("<h3 class='tab-header'>Model Performance Details</h3>", unsafe_allow_html=True)
            
            if is_classification:
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
                ax.set_title("Confusion Matrix")
                ax.set_ylabel("True Label")
                ax.set_xlabel("Predicted Label")
                st.pyplot(fig, use_container_width=True)

                st.text("Classification Report:")
                st.code(classification_report(y_test, y_pred))
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    residuals = y_test - y_pred
                    fig = px.histogram(residuals, nbins=30, title="Residuals Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        x=y_test,
                        y=y_pred,
                        title="Actual vs Predicted",
                        labels={"x": "Actual", "y": "Predicted"}
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # -------- TAB 4: SHAP EXPLANATION --------
        with tabs[3]:
            st.markdown("<h3 class='tab-header'>Model Explainability (SHAP)</h3>", unsafe_allow_html=True)
            
            if st.button("🔍 Generate SHAP Explanation"):
                sample = X_train.sample(min(SAMPLE_SIZE, len(X_train)))
                create_shap_explanation(pipeline, sample)

        # -------- TAB 5: CLUSTERING --------
        with tabs[4]:
            st.markdown("<h3 class='tab-header'>Student Segmentation via Clustering</h3>", unsafe_allow_html=True)

            if len(numeric_features) >= 2:
                k = st.slider("Number of Clusters", MIN_CLUSTERS, MAX_CLUSTERS, DEFAULT_CLUSTERS)

                # Prepare data
                cluster_data = df[numeric_features].dropna().copy()

                # Apply KMeans
                try:
                    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
                    cluster_data["Cluster"] = kmeans.fit_predict(cluster_data)

                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        x_axis = st.selectbox("X-Axis Feature", numeric_features, key="cluster_x")
                    with col2:
                        y_axis = st.selectbox("Y-Axis Feature", numeric_features, key="cluster_y")

                    fig = px.scatter(
                        cluster_data,
                        x=x_axis,
                        y=y_axis,
                        color="Cluster",
                        title=f"Student Clusters (k={k})",
                        size_max=10,
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Cluster summary
                    st.markdown("### 📋 Cluster Statistics")
                    summary = cluster_data.groupby("Cluster").mean().round(2)
                    st.dataframe(summary, use_container_width=True)

                    # Cluster interpretation
                    st.markdown("### 🎯 Cluster Interpretation")
                    important_feature = numeric_features[0]
                    best_cluster = summary[important_feature].idxmax()
                    worst_cluster = summary[important_feature].idxmin()

                    col1, col2, col3 = st.columns(3)
                    
                    for cluster_id in sorted(summary.index):
                        if cluster_id == best_cluster:
                            with col1:
                                st.success(f"✅ Cluster {cluster_id}: High Performing")
                        elif cluster_id == worst_cluster:
                            with col2:
                                st.error(f"❌ Cluster {cluster_id}: At-Risk")
                        else:
                            with col3:
                                st.warning(f"⚠️ Cluster {cluster_id}: Average")

                    # Cluster distribution
                    st.markdown("### 📊 Students per Cluster")
                    counts = cluster_data["Cluster"].value_counts().sort_index()
                    fig = px.bar(counts, title="Cluster Distribution", labels={"index": "Cluster", "value": "Count"})
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Clustering error: {str(e)}")
            else:
                st.warning("⚠️ Need at least 2 numeric features for clustering")

        # -------- TAB 6: DATA VIEW --------
        with tabs[5]:
            st.markdown("<h3 class='tab-header'>Data Overview</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", len(df))
            col2.metric("Total Columns", len(df.columns))
            col3.metric("Missing Values", df.isnull().sum().sum())

            st.markdown("### Missing Values")
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                st.bar_chart(missing_data[missing_data > 0])
            else:
                st.success("✅ No missing values!")

            st.markdown("### Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

        # ============== DOWNLOAD RESULTS ==============
        st.divider()
        st.markdown("### 📥 Export Results")
        
        df_export = df.copy()
        df_export["Prediction"] = pipeline.predict(X)
        df_export["Risk_Level"] = df_export["Prediction"].apply(lambda x: classify_risk(x, is_classification))

        csv = df_export.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="student_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted.")
else:
    st.info("📤 **To get started:** Upload a CSV file in the sidebar to begin analyzing student data.")
    
    # Show sample data structure
    with st.expander("📋 Expected Data Format"):
        sample_data = pd.DataFrame({
            "Student_Name": ["John", "Jane", "Bob"],
            "Student_ID": [101, 102, 103],
            "Math": [85, 90, 75],
            "Science": [88, 92, 78],
            "English": [80, 85, 82],
            "Performance": ["Pass", "Pass", "Fail"]
        })
        st.dataframe(sample_data, use_container_width=True)
