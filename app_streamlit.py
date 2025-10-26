"""
🔬 Classification Models Comparison - Interactive Dashboard
Streamlit web app for visualizing and comparing machine learning models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

# Page config
st.set_page_config(
    page_title="ML Models Comparison Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_results(dataset='cardio_train'):
    """Load the latest results CSV file"""
    results_dir = Path(f"experiments/full_comparison/{dataset}")

    # Try multiple patterns
    csv_files = list(results_dir.glob("full_comparison_*.csv"))
    if not csv_files:
        csv_files = list(results_dir.glob(f"{dataset}_*.csv"))
    if not csv_files:
        csv_files = list(results_dir.glob("*.csv"))

    if not csv_files:
        return None, None

    # Get the latest file
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)

    return df, latest_file.name

@st.cache_data
def load_best_model_metadata():
    """Load best model metadata"""
    metadata_dir = Path("experiments/full_comparison/best_model")
    json_files = list(metadata_dir.glob("metadata_*.json"))
    
    if not json_files:
        return None
    
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    with open(latest_file, 'r') as f:
        return json.load(f)

def create_metrics_comparison(df, selected_models):
    """Create interactive metrics comparison chart"""
    metrics = ['pr_auc', 'sensitivity', 'specificity', 'f1', 'roc_auc', 'mcc']
    
    filtered_df = df[df['model'].isin(selected_models)]
    
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric.upper().replace('_', '-'),
            x=filtered_df['model'],
            y=filtered_df[metric],
            text=filtered_df[metric].round(4),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Metrics Comparison Across Models",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_generation_comparison(df):
    """Create generation comparison visualization"""
    gen_stats = df.groupby('generation').agg({
        'pr_auc': ['mean', 'std', 'max', 'min'],
        'sensitivity': 'mean',
        'specificity': 'mean',
        'f1': 'mean'
    }).round(4)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PR-AUC Statistics', 'Average Sensitivity', 
                       'Average Specificity', 'Average F1-Score'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    generations = gen_stats.index.tolist()
    
    # PR-AUC with error bars
    fig.add_trace(
        go.Bar(
            x=generations,
            y=gen_stats[('pr_auc', 'mean')],
            error_y=dict(type='data', array=gen_stats[('pr_auc', 'std')]),
            name='PR-AUC',
            marker_color='lightblue',
            text=gen_stats[('pr_auc', 'mean')].round(4),
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Sensitivity
    fig.add_trace(
        go.Bar(
            x=generations,
            y=gen_stats[('sensitivity', 'mean')],
            name='Sensitivity',
            marker_color='lightgreen',
            text=gen_stats[('sensitivity', 'mean')].round(4),
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Specificity
    fig.add_trace(
        go.Bar(
            x=generations,
            y=gen_stats[('specificity', 'mean')],
            name='Specificity',
            marker_color='lightcoral',
            text=gen_stats[('specificity', 'mean')].round(4),
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # F1-Score
    fig.add_trace(
        go.Bar(
            x=generations,
            y=gen_stats[('f1', 'mean')],
            name='F1-Score',
            marker_color='lightyellow',
            text=gen_stats[('f1', 'mean')].round(4),
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def create_preprocessing_impact(df):
    """Analyze preprocessing impact on performance"""
    # PR-AUC by preprocessing config
    prep_impact = df.groupby(['scaler', 'imbalance', 'feature_selection'])['pr_auc'].mean().reset_index()
    prep_impact = prep_impact.sort_values('pr_auc', ascending=False).head(15)
    
    prep_impact['config'] = (prep_impact['scaler'].fillna('None') + ' | ' + 
                             prep_impact['imbalance'].fillna('None') + ' | ' + 
                             prep_impact['feature_selection'].fillna('None'))
    
    fig = px.bar(
        prep_impact,
        x='pr_auc',
        y='config',
        orientation='h',
        title='Top 15 Preprocessing Configurations by PR-AUC',
        labels={'pr_auc': 'PR-AUC', 'config': 'Preprocessing Config'},
        text='pr_auc',
        color='pr_auc',
        color_continuous_scale='Viridis'
    )
    
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(height=600, template='plotly_white')
    
    return fig

def create_scatter_matrix(df, selected_models):
    """Create scatter matrix for key metrics"""
    metrics = ['pr_auc', 'sensitivity', 'specificity', 'f1', 'roc_auc']
    
    filtered_df = df[df['model'].isin(selected_models)][metrics + ['model']]
    
    fig = px.scatter_matrix(
        filtered_df,
        dimensions=metrics,
        color='model',
        title='Metrics Correlation Matrix',
        height=800,
        labels={col: col.upper().replace('_', '-') for col in metrics}
    )
    
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(template='plotly_white')
    
    return fig

def create_training_time_analysis(df):
    """Analyze training time vs performance"""
    fig = px.scatter(
        df,
        x='train_time_sec',
        y='pr_auc',
        color='generation',
        size='train_time_sec',
        hover_data=['model', 'sensitivity', 'specificity'],
        title='Performance vs Training Time',
        labels={
            'train_time_sec': 'Training Time (seconds)',
            'pr_auc': 'PR-AUC',
            'generation': 'Generation'
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(height=500, template='plotly_white')
    
    return fig

def create_radar_chart(df, selected_models):
    """Create radar chart for top models"""
    metrics = ['pr_auc', 'sensitivity', 'specificity', 'f1', 'roc_auc', 'mcc']
    
    filtered_df = df[df['model'].isin(selected_models)]
    
    fig = go.Figure()
    
    for idx, row in filtered_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics],
            theta=[m.upper().replace('_', '-') for m in metrics],
            fill='toself',
            name=row['model']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='Multi-Metric Performance Comparison',
        height=600,
        template='plotly_white'
    )
    
    return fig

def create_heatmap(df):
    """Create correlation heatmap for metrics"""
    metrics = ['pr_auc', 'sensitivity', 'specificity', 'f1', 'roc_auc', 'mcc', 'npv']
    corr_matrix = df[metrics].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.3f',
        aspect='auto',
        title='Metrics Correlation Heatmap',
        color_continuous_scale='RdBu_r',
        labels=dict(color="Correlation")
    )
    
    fig.update_layout(height=600, template='plotly_white')
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 ML Classification: Performance Insights</h1>',
                unsafe_allow_html=True)

    # Info banner
    st.info("""
    👋 **Chào mừng đến với Dashboard Phân Tích Models!**

    📄 **Trang mới:** Xem [Experimental Setup](./Experimental_Setup) để hiểu về datasets, models và configs được sử dụng.
    """)

    # Sidebar - Dataset selection first
    st.sidebar.title("⚙️ Configuration")

    # Dataset selection
    dataset_choice = st.sidebar.radio(
        "📊 Dataset",
        options=['cardio_train', 'creditcard'],
        format_func=lambda x: "❤️ Cardio Train" if x == 'cardio_train' else "💳 Credit Card",
        index=0
    )

    # Load data
    df, filename = load_results(dataset=dataset_choice)

    if df is None:
        st.error(f"❌ No results found for {dataset_choice}!")
        st.info("Run: `python full_comparison.py --data data/raw/{dataset_choice}.csv`")
        return

    # Show dataset info in sidebar
    st.sidebar.success(f"✅ {len(df)} experiments loaded")
    
    st.sidebar.markdown("---")
    
    # Primary metric selection at top
    primary_metric = st.sidebar.selectbox(
        "🎯 Focus Metric",
        options=['f1', 'accuracy', 'balanced_accuracy', 'pr_auc', 'roc_auc', 'sensitivity', 'specificity', 'mcc'],
        index=0,
        format_func=lambda x: {
            'f1': '🎯 F1 Score',
            'accuracy': '✅ Accuracy',
            'balanced_accuracy': '⚖️ Balanced Accuracy',
            'pr_auc': '� PR-AUC', 
            'roc_auc': '📈 ROC-AUC',
            'sensitivity': '🔍 Sensitivity',
            'specificity': '🛡️ Specificity',
            'mcc': '🔢 MCC'
        }.get(x, x)
    )
    
    # Show metric description
    metric_descriptions = {
        'f1': """
**F1 Score** - Harmonic mean of Precision (accuracy when predicting positive) and Recall (% of actual positives found). Use when you want to balance "not missing cases" vs "not false alarming". Works best with balanced datasets. F1 > 0.7 is good, > 0.9 is excellent.
""",
        'accuracy': """
**Accuracy** - Proportion of correct predictions overall. Simplest metric but MISLEADING with imbalanced data! Example: Predicting "no fraud" for all cases gives 99% accuracy when fraud is 1%, but misses 100% of fraud. Only use for balanced datasets (50/50).
""",
        'balanced_accuracy': """
**Balanced Accuracy** - Average of per-class accuracy. Better than Accuracy for imbalanced data because it's FAIR to both classes. A lazy model predicting all one class gets 50%, not 99% like Accuracy. Always prefer this over Accuracy when data is imbalanced.
""",
        'pr_auc': """
**PR-AUC** - Area under Precision-Recall curve. Most HONEST metric for SEVERELY IMBALANCED data (0.1% fraud, rare disease...). ROC-AUC can be 0.99 while PR-AUC = 0.3 REVEALS the model is actually bad. PR-AUC > 0.5 beats random, > 0.7 is good. For imbalanced data, PR-AUC is MORE IMPORTANT than ROC-AUC!
""",
        'roc_auc': """
**ROC-AUC** - Measures ability to distinguish between 2 classes. 0.5 = random (coin flip), 0.7-0.8 = good, > 0.9 = excellent. Works well with balanced datasets. CAUTION: High ROC-AUC doesn't guarantee good model on imbalanced data (can be deceiving). Use PR-AUC instead for imbalanced cases.
""",
        'sensitivity': """
**Sensitivity (Recall)** - Proportion of actual positives correctly found (% of patients detected). Critical in medical/security where MISSING CASES = DANGEROUS. Example: 100 COVID cases, test finds 95 → Sensitivity = 95%. Medical/security needs > 95%. Trade-off: High Sensitivity → fewer misses but more false alarms.
""",
        'specificity': """
**Specificity** - Proportion of actual negatives correctly identified (% of healthy correctly confirmed). Important when you don't want false positives. Example: Email spam filter with high Specificity → fewer important emails blocked. Trade-off: High Specificity → fewer false alarms but may miss positives. Balance with Sensitivity based on problem.
""",
        'mcc': """
**MCC** - ONLY metric that's accurate for IMBALANCED data, considers all 4 outcomes (TP/TN/FP/FN). Values: +1 = perfect, 0 = random, -1 = worse than random. Lazy model predicting all one class: Accuracy 99% but MCC = 0 (useless!). MCC < 0.3 = poor, 0.5-0.7 = good, > 0.7 = very good. Trusted in research.
"""
    }

    with st.sidebar.expander("ℹ️ About This Metric", expanded=False):
        st.markdown(metric_descriptions.get(primary_metric, ""))
    
    st.sidebar.markdown("---")
    
    # Quick filters
    top_n = st.sidebar.slider("📌 Show Top N Models", 5, 20, 10)
    
    # Main content - EXECUTIVE SUMMARY FIRST
    st.markdown("## 🎯 Executive Summary")
    
    # KEY METRICS ROW
    col1, col2, col3, col4, col5 = st.columns(5)
    
    best_idx = df[primary_metric].idxmax()
    best_model = df.loc[best_idx]
    avg_score = df[primary_metric].mean()
    
    with col1:
        st.metric(
            "🏆 Best Score",
            f"{best_model[primary_metric]:.3f}",
            delta=f"+{(best_model[primary_metric] - avg_score):.3f} vs avg",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "📊 Average",
            f"{avg_score:.3f}",
            help=f"Average {primary_metric} across all {len(df)} experiments"
        )
    
    with col3:
        st.metric(
            "🎰 Variance",
            f"{df[primary_metric].std():.3f}",
            help="Standard deviation - lower is more consistent"
        )
    
    with col4:
        st.metric(
            "⚡ Best Time",
            f"{best_model['train_time_sec']:.1f}s",
            help="Training time of best model"
        )
    
    with col5:
        st.metric(
            "🔢 Experiments",
            f"{len(df)}",
            help="Total configurations tested"
        )
    
    st.markdown("---")
    
    # WINNER ANNOUNCEMENT
    st.markdown("### 🏆 Champion Configuration")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown(f"""
        **Model:** `{best_model['model']}`  
        **Generation:** `{best_model['generation']}`  
        **Performance:** `{best_model[primary_metric]:.4f}`
        """)
    
    with col2:
        st.markdown(f"""
        **Scaler:** `{best_model.get('scaler', 'none')}`  
        **Imbalance:** `{best_model.get('imbalance', 'none')}`  
        **Features:** `{best_model.get('feature_selection', 'all')}`
        """)
    
    with col3:
        # Calculate efficiency
        efficiency = best_model[primary_metric] / (best_model['train_time_sec'] + 1)
        st.metric("Efficiency", f"{efficiency:.4f}", help="Score per second")
    
    st.markdown("---")
    
    # VISUAL COMPARISON - The money shot!
    st.markdown("### 📊 Top Performers Comparison")
    
    top_models = df.nlargest(top_n, primary_metric)
    
    # Create comprehensive comparison chart
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Top {top_n} Models by {primary_metric.upper()}', 
                       'Multi-Metric Radar'),
        specs=[[{'type': 'bar'}, {'type': 'scatterpolar'}]],
        column_widths=[0.6, 0.4]
    )
    
    # Bar chart
    fig.add_trace(
        go.Bar(
            x=top_models['model'],
            y=top_models[primary_metric],
            text=top_models[primary_metric].round(4),
            textposition='outside',
            marker=dict(
                color=top_models[primary_metric],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(x=0.45)
            ),
            hovertemplate='<b>%{x}</b><br>' +
                         f'{primary_metric.upper()}: %{{y:.4f}}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Radar for top 5 - include more metrics
    metrics = ['accuracy', 'f1', 'sensitivity', 'specificity', 'pr_auc', 'roc_auc']
    # Filter to available metrics
    metrics = [m for m in metrics if m in df.columns]
    
    for idx, row in top_models.head(5).iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[row[m] for m in metrics],
                theta=[m.upper() for m in metrics],
                name=row['model'][:20],  # Truncate name
                fill='toself'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        template='plotly_white',
        polar=dict(radialaxis=dict(visible=True, range=[0, 1]))
    )
    
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # PREPROCESSING IMPACT - Critical insights!
    st.markdown("### 🔧 What Really Matters: Config Impact Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📐 Scaler Effect")
        scaler_impact = df.groupby('scaler').agg({
            primary_metric: ['mean', 'count']
        }).round(4)
        scaler_impact.columns = ['Avg Score', 'Count']
        scaler_impact = scaler_impact.sort_values('Avg Score', ascending=False)
        
        # Visual indicator
        best_scaler = scaler_impact.index[0]
        worst_scaler = scaler_impact.index[-1]
        improvement = scaler_impact.loc[best_scaler, 'Avg Score'] - scaler_impact.loc[worst_scaler, 'Avg Score']
        
        st.dataframe(scaler_impact, width='stretch')
        
        if improvement > 0.05:
            st.success(f"✅ {best_scaler}: +{improvement:.3f} improvement!")
        elif improvement > 0.01:
            st.info(f"ℹ️ {best_scaler}: +{improvement:.3f} better")
        else:
            st.warning(f"⚠️ Scaler choice: minimal impact ({improvement:.3f})")
    
    with col2:
        st.markdown("#### ⚖️ Imbalance Handling")
        imb_impact = df.groupby('imbalance').agg({
            primary_metric: ['mean', 'count']
        }).round(4)
        imb_impact.columns = ['Avg Score', 'Count']
        imb_impact = imb_impact.sort_values('Avg Score', ascending=False)
        
        best_imb = imb_impact.index[0]
        worst_imb = imb_impact.index[-1]
        improvement = imb_impact.loc[best_imb, 'Avg Score'] - imb_impact.loc[worst_imb, 'Avg Score']
        
        st.dataframe(imb_impact, width='stretch')
        
        if improvement > 0.05:
            st.success(f"✅ {best_imb}: +{improvement:.3f} improvement!")
        elif improvement > 0.01:
            st.info(f"ℹ️ {best_imb}: +{improvement:.3f} better")
        else:
            st.warning(f"⚠️ Imbalance: minimal impact ({improvement:.3f})")
    
    with col3:
        st.markdown("#### 🎯 Feature Selection")
        fs_impact = df.groupby('feature_selection').agg({
            primary_metric: ['mean', 'count']
        }).round(4)
        fs_impact.columns = ['Avg Score', 'Count']
        fs_impact = fs_impact.sort_values('Avg Score', ascending=False)
        
        best_fs = fs_impact.index[0]
        worst_fs = fs_impact.index[-1]
        improvement = fs_impact.loc[best_fs, 'Avg Score'] - fs_impact.loc[worst_fs, 'Avg Score']
        
        st.dataframe(fs_impact, width='stretch')
        
        if improvement > 0.05:
            st.success(f"✅ {best_fs}: +{improvement:.3f} improvement!")
        elif improvement > 0.01:
            st.info(f"ℹ️ {best_fs}: +{improvement:.3f} better")
        else:
            st.warning(f"⚠️ Features: minimal impact ({improvement:.3f})")
    
    st.markdown("---")
    
    # MODEL FAMILY COMPARISON
    st.markdown("### 🏗️ Model Architecture Comparison")
    
    # Extract model family (remove generation prefix)
    df['model_family'] = df['model'].str.replace(r'^Gen\d+_', '', regex=True)
    
    family_stats = df.groupby('model_family').agg({
        primary_metric: ['mean', 'std', 'max', 'count']
    }).round(4)
    family_stats.columns = ['Mean', 'Std', 'Max', 'Count']
    family_stats = family_stats.sort_values('Mean', ascending=False).head(10)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Average Score',
        x=family_stats.index,
        y=family_stats['Mean'],
        error_y=dict(type='data', array=family_stats['Std']),
        text=family_stats['Mean'].round(3),
        textposition='outside',
        marker=dict(color=family_stats['Mean'], colorscale='Blues')
    ))
    
    fig.update_layout(
        title=f'Model Family Performance (Top 10 by {primary_metric.upper()})',
        xaxis_title='Model Type',
        yaxis_title=f'{primary_metric.upper()} Score',
        height=400,
        template='plotly_white',
        xaxis={'tickangle': -45}
    )
    
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # EFFICIENCY ANALYSIS
    st.markdown("### ⚡ Performance vs Speed Trade-off")
    
    df['efficiency'] = df[primary_metric] / (df['train_time_sec'] + 0.1)
    
    fig = px.scatter(
        df,
        x='train_time_sec',
        y=primary_metric,
        color='generation',
        size=df[primary_metric],
        hover_data=['model', 'scaler', 'imbalance'],
        labels={
            'train_time_sec': 'Training Time (seconds)',
            primary_metric: f'{primary_metric.upper()} Score'
        },
        title='Performance vs Training Time (bubble size = score)',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(height=450, template='plotly_white')
    st.plotly_chart(fig, width='stretch')
    
    # Top efficient models
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚀 Most Efficient (Score/Time)")
        top_efficient = df.nlargest(5, 'efficiency')[['model', primary_metric, 'train_time_sec', 'efficiency']]
        top_efficient.columns = ['Model', 'Score', 'Time(s)', 'Efficiency']
        st.dataframe(top_efficient.round(4), width='stretch', hide_index=True)
    
    with col2:
        st.markdown("#### 🎯 Best Raw Performance")
        top_performance = df.nlargest(5, primary_metric)[['model', primary_metric, 'train_time_sec', 'efficiency']]
        top_performance.columns = ['Model', 'Score', 'Time(s)', 'Efficiency']
        st.dataframe(top_performance.round(4), width='stretch', hide_index=True)
    
    st.markdown("---")
    
    # DETAILED TABS - For deeper exploration
    with st.expander("🔍 Detailed Analysis & Raw Data", expanded=False):
        
        tab1, tab2, tab3 = st.tabs(["📋 Full Results", "🎨 Heatmaps", "📥 Export"])
        
        with tab1:
            st.markdown("##### All Experiment Results")
            
            # Sorting controls
            sort_col1, sort_col2 = st.columns([3, 1])
            with sort_col1:
                sort_by = st.selectbox("Sort by", options=[primary_metric, 'train_time_sec', 'efficiency', 'model'])
            with sort_col2:
                ascending = st.checkbox("Ascending", value=False)
            
            sorted_df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
            
            # Build display columns list, ensuring no duplicates
            display_cols = ['model', 'generation', primary_metric, 'accuracy', 'balanced_accuracy',
                           'sensitivity', 'specificity', 'precision', 'f1', 'pr_auc', 'roc_auc', 
                           'mcc', 'train_time_sec', 'scaler', 'imbalance', 'feature_selection']
            
            # Remove duplicates while preserving order
            display_cols = list(dict.fromkeys(display_cols))
            
            # Filter columns that exist
            display_cols = [col for col in display_cols if col in sorted_df.columns]
            
            # Create a copy with unique index to avoid styling issues
            display_df = sorted_df[display_cols].copy()
            display_df.index = range(len(display_df))
            
            # Format numeric columns
            format_dict = {}
            for col in ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'precision', 
                       'f1', 'pr_auc', 'roc_auc', 'mcc']:
                if col in display_df.columns:
                    format_dict[col] = '{:.4f}'
            if 'train_time_sec' in display_df.columns:
                format_dict['train_time_sec'] = '{:.2f}'
            
            # Display with styling only if safe
            try:
                # Get metrics that exist in display_df for styling
                style_cols = [col for col in [primary_metric, 'accuracy', 'f1', 'sensitivity', 'specificity', 'pr_auc'] 
                             if col in display_df.columns]
                style_cols = list(dict.fromkeys(style_cols))  # Remove duplicates
                
                styled_df = display_df.style.background_gradient(
                    subset=style_cols,
                    cmap='RdYlGn'
                ).format(format_dict)
                
                st.dataframe(styled_df, width='stretch', height=400)
            except Exception as e:
                # If styling fails, just show plain dataframe
                st.warning(f"Note: Styling disabled due to data structure. Showing plain table.")
                st.dataframe(display_df, width='stretch', height=400)
        
        with tab2:
            st.markdown("##### Configuration Interaction Heatmaps")
            
            # Scaler x Imbalance
            pivot_scaler_imb = df.pivot_table(
                values=primary_metric,
                index='scaler',
                columns='imbalance',
                aggfunc='mean'
            )
            
            fig = px.imshow(
                pivot_scaler_imb,
                text_auto='.3f',
                aspect='auto',
                title='Scaler × Imbalance Interaction',
                color_continuous_scale='RdYlGn',
                labels=dict(color=primary_metric.upper())
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
            
            # Model x Feature Selection
            df_top_models = df[df['model'].isin(df.groupby('model')[primary_metric].mean().nlargest(10).index)]
            pivot_model_fs = df_top_models.pivot_table(
                values=primary_metric,
                index='model',
                columns='feature_selection',
                aggfunc='mean'
            )
            
            fig = px.imshow(
                pivot_model_fs,
                text_auto='.3f',
                aspect='auto',
                title='Top 10 Models × Feature Selection',
                color_continuous_scale='Viridis',
                labels=dict(color=primary_metric.upper())
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width='stretch')
        
        with tab3:
            st.markdown("##### Export Results")
            
            # Download top models
            csv_top = top_models.to_csv(index=False)
            st.download_button(
                label=f"📥 Download Top {top_n} Models CSV",
                data=csv_top,
                file_name=f"top_{top_n}_models_{dataset_choice}.csv",
                mime="text/csv"
            )
            
            # Download all
            csv_all = df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Results CSV",
                data=csv_all,
                file_name=f"all_results_{dataset_choice}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.caption(f"💾 Data: {filename} | 🔢 {len(df)} experiments | 🎯 Focused on {primary_metric.upper()}")

if __name__ == "__main__":
    main()
