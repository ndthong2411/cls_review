"""
Streamlit Demo App - Progressive Model Comparison for CVD Prediction

This app allows you to:
1. Load and explore the Kaggle CVD dataset
2. Configure preprocessing and models
3. Train and compare multiple models
4. Visualize results across generations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from pathlib import Path
import json

# Page config
st.set_page_config(
    page_title="CVD Model Comparison",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("❤️ Cardiovascular Disease Prediction - Progressive Model Comparison")
st.markdown("""
This demo compares machine learning models across 3 generations:
- **Generation 1 (Baseline)**: Logistic Regression, Decision Tree, KNN
- **Generation 2 (Intermediate)**: Random Forest, SVM, Gradient Boosting  
- **Generation 3 (Advanced)**: XGBoost, LightGBM, CatBoost, MLP
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Data loading
@st.cache_data
def load_data():
    """Load the dataset"""
    data_path = Path("data/raw/cardio_train.csv")
    if not data_path.exists():
        st.error(f"Dataset not found at {data_path}. Please download from Kaggle.")
        return None
    
    df = pd.read_csv(data_path, sep=';')
    df.columns = df.columns.str.strip()
    
    # Feature engineering
    df['age_years'] = df['age'] / 365.25
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
    
    return df

# Load results if available
@st.cache_data
def load_results():
    """Load experiment results"""
    results_path = Path("experiments/results_summary.csv")
    if results_path.exists():
        return pd.read_csv(results_path)
    return None

# Main app sections
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Explorer", 
    "🎯 Model Training", 
    "📈 Results Comparison",
    "🔮 Make Predictions"
])

# TAB 1: Data Explorer
with tab1:
    st.header("Dataset Overview")
    
    df = load_data()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", f"{len(df):,}")
        with col2:
            st.metric("Features", f"{len(df.columns)-1}")
        with col3:
            positive_rate = (df['cardio'].sum() / len(df) * 100)
            st.metric("Positive Rate", f"{positive_rate:.1f}%")
        with col4:
            imbalance = df['cardio'].value_counts()
            ratio = imbalance.max() / imbalance.min()
            st.metric("Imbalance Ratio", f"{ratio:.2f}:1")
        
        # Sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Target distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Target Distribution")
            target_dist = df['cardio'].value_counts()
            fig = px.pie(
                values=target_dist.values,
                names=['No CVD', 'CVD'],
                title="Cardiovascular Disease Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Age Distribution by Target")
            fig = px.histogram(
                df,
                x='age_years',
                color='cardio',
                nbins=30,
                title="Age Distribution",
                labels={'cardio': 'CVD Status'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlations
        st.subheader("Feature Correlation Heatmap")
        numeric_cols = ['age_years', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo', 
                       'pulse_pressure', 'map', 'cholesterol', 'gluc', 'cardio']
        corr = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: Model Training
with tab2:
    st.header("Train and Compare Models")
    
    st.info("⚠️ **Note**: Full training pipeline requires running the complete experiment script. "
            "This demo shows configuration options.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Preprocessing Options")
        
        missing_strategy = st.selectbox(
            "Missing Value Handling",
            ['median', 'mean', 'knn', 'mice'],
            help="How to handle missing values"
        )
        
        outlier_method = st.selectbox(
            "Outlier Detection",
            ['none', 'iqr_clip', 'zscore_clip'],
            help="How to handle outliers"
        )
        
        scaler = st.selectbox(
            "Feature Scaling",
            ['standard', 'robust', 'minmax'],
            help="Scaling method for numerical features"
        )
        
        imbalance_method = st.selectbox(
            "Imbalance Handling",
            ['class_weight', 'smote', 'smoteenn', 'adasyn'],
            help="How to handle class imbalance"
        )
    
    with col2:
        st.subheader("Model Selection")
        
        generation = st.radio(
            "Model Generation",
            ['Generation 1 (Baseline)', 'Generation 2 (Intermediate)', 'Generation 3 (Advanced)']
        )
        
        if 'Generation 1' in generation:
            models = st.multiselect(
                "Select Models",
                ['Logistic Regression', 'Decision Tree', 'KNN'],
                default=['Logistic Regression']
            )
        elif 'Generation 2' in generation:
            models = st.multiselect(
                "Select Models",
                ['Random Forest', 'SVM', 'Gradient Boosting'],
                default=['Random Forest']
            )
        else:
            models = st.multiselect(
                "Select Models",
                ['XGBoost', 'LightGBM', 'CatBoost', 'MLP'],
                default=['XGBoost']
            )
        
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        
        use_optuna = st.checkbox("Use Optuna for Hyperparameter Tuning", value=False)
        if use_optuna:
            n_trials = st.number_input("Number of Trials", 10, 200, 50)
    
    # Training command
    st.subheader("Training Command")
    phase = 'baseline' if 'Generation 1' in generation else ('intermediate' if 'Generation 2' in generation else 'advanced')
    
    command = f"""python -m src.experiment.run_phase \\
    --phase={phase} \\
    preprocessing.missing={missing_strategy} \\
    preprocessing.outliers={outlier_method} \\
    preprocessing.scale={scaler} \\
    imbalance.method={imbalance_method} \\
    cv.n_splits={cv_folds}"""
    
    st.code(command, language='bash')
    
    if st.button("🚀 Start Training (Coming Soon)", disabled=True):
        st.info("Training functionality will be available after running the full pipeline setup.")

# TAB 3: Results Comparison
with tab3:
    st.header("Model Performance Comparison")
    
    results = load_results()
    
    if results is not None:
        st.success(f"Loaded {len(results)} experiment results")
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_gens = st.multiselect(
                "Filter by Generation",
                options=results['generation'].unique() if 'generation' in results.columns else [],
                default=results['generation'].unique() if 'generation' in results.columns else []
            )
        
        with col2:
            metric_to_plot = st.selectbox(
                "Primary Metric",
                ['pr_auc', 'roc_auc', 'f1', 'recall', 'precision']
            )
        
        with col3:
            top_n = st.slider("Show Top N Models", 5, 20, 10)
        
        # Filter results
        if selected_gens:
            filtered_results = results[results['generation'].isin(selected_gens)]
        else:
            filtered_results = results
        
        # Top models
        top_models = filtered_results.nlargest(top_n, metric_to_plot)
        
        # Performance comparison
        st.subheader(f"Top {top_n} Models by {metric_to_plot.upper()}")
        
        fig = px.bar(
            top_models,
            x='model_name',
            y=metric_to_plot,
            color='generation',
            title=f"{metric_to_plot.upper()} Comparison",
            labels={'model_name': 'Model', metric_to_plot: metric_to_plot.upper()}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Generation comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance by Generation")
            gen_avg = filtered_results.groupby('generation')[metric_to_plot].mean().reset_index()
            fig = px.bar(
                gen_avg,
                x='generation',
                y=metric_to_plot,
                title=f"Average {metric_to_plot.upper()} by Generation"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Training Time vs Performance")
            fig = px.scatter(
                filtered_results,
                x='train_time_sec',
                y=metric_to_plot,
                color='generation',
                size=metric_to_plot,
                hover_data=['model_name'],
                title="Efficiency Frontier"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("Detailed Results")
        display_cols = ['model_name', 'generation', 'recall', 'precision', 'f1', 
                       'pr_auc', 'roc_auc', 'train_time_sec']
        display_cols = [col for col in display_cols if col in filtered_results.columns]
        st.dataframe(
            filtered_results[display_cols].sort_values(metric_to_plot, ascending=False),
            use_container_width=True
        )
    
    else:
        st.warning("No results found. Please run experiments first.")
        st.info("""
        To generate results, run:
        ```bash
        python -m src.experiment.run_phase --phase=baseline
        python -m src.experiment.run_phase --phase=intermediate  
        python -m src.experiment.run_phase --phase=advanced
        ```
        """)

# TAB 4: Make Predictions
with tab4:
    st.header("Make Predictions with Trained Models")
    
    # Check for saved models
    models_dir = Path("experiments/models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl"))
        
        if model_files:
            selected_model = st.selectbox(
                "Select Trained Model",
                options=[f.stem for f in model_files]
            )
            
            st.subheader("Patient Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age (years)", 20, 100, 50)
                gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
                height = st.number_input("Height (cm)", 120, 220, 170)
                weight = st.number_input("Weight (kg)", 40, 200, 70)
            
            with col2:
                ap_hi = st.number_input("Systolic BP", 80, 250, 120)
                ap_lo = st.number_input("Diastolic BP", 50, 150, 80)
                cholesterol = st.selectbox("Cholesterol", [1, 2, 3], 
                                         format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
                gluc = st.selectbox("Glucose", [1, 2, 3],
                                   format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
            
            with col3:
                smoke = st.checkbox("Smoker")
                alco = st.checkbox("Alcohol Consumption")
                active = st.checkbox("Physically Active")
            
            if st.button("🔮 Predict"):
                st.info("Prediction functionality requires loading the saved model and pipeline. "
                       "This will be available after running the full experiment.")
        else:
            st.warning("No trained models found in experiments/models/")
    else:
        st.warning("Models directory not found. Please train models first.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Progressive Model Comparison Pipeline for Cardiovascular Disease Prediction</p>
    <p>Built with Streamlit • PyTorch • scikit-learn • XGBoost</p>
</div>
""", unsafe_allow_html=True)
