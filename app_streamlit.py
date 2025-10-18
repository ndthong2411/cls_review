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
    st.markdown('<h1 class="main-header">🔬 ML Models Comparison Dashboard</h1>',
                unsafe_allow_html=True)

    # Sidebar - Dataset selection first
    st.sidebar.title("⚙️ Settings")

    # Dataset selection
    st.sidebar.header("📁 Dataset")
    dataset_choice = st.sidebar.radio(
        "Select Dataset",
        options=['cardio_train', 'creditcard'],
        index=0
    )

    # Load data
    df, filename = load_results(dataset=dataset_choice)

    if df is None:
        st.error(f"❌ No results found for {dataset_choice}! Please run `python full_comparison.py --data data/raw/{dataset_choice}.csv` first.")
        return

    st.success(f"✅ Loaded results from: `{filename}`")
    
    # Model selection
    st.sidebar.header("📊 Model Selection")
    
    all_models = df['model'].unique().tolist()
    generations = sorted(df['generation'].unique())
    
    # Generation filter
    selected_gen = st.sidebar.multiselect(
        "Filter by Generation",
        options=generations,
        default=generations
    )
    
    filtered_models = df[df['generation'].isin(selected_gen)]['model'].unique().tolist()
    
    # Model multiselect
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        options=filtered_models,
        default=filtered_models[:5] if len(filtered_models) >= 5 else filtered_models
    )
    
    # Metric selection
    st.sidebar.header("📈 Primary Metric")
    primary_metric = st.sidebar.selectbox(
        "Select Primary Metric",
        options=['pr_auc', 'sensitivity', 'specificity', 'f1', 'roc_auc', 'mcc'],
        format_func=lambda x: x.upper().replace('_', '-')
    )
    
    # Top N filter
    top_n = st.sidebar.slider("Show Top N Models", 5, 50, 10)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🏆 Top Models", 
        "📈 Generation Analysis",
        "🔧 Preprocessing Impact",
        "⏱️ Performance Analysis",
        "📋 Detailed Data"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Experiments",
                len(df),
                help="Total number of model configurations tested"
            )
        
        with col2:
            st.metric(
                "Unique Models",
                df['model'].nunique(),
                help="Number of unique model types"
            )
        
        with col3:
            st.metric(
                "Generations",
                df['generation'].nunique(),
                help="Number of model generations"
            )
        
        with col4:
            best_score = df[primary_metric].max()
            st.metric(
                f"Best {primary_metric.upper().replace('_', '-')}",
                f"{best_score:.4f}",
                help=f"Highest {primary_metric} score achieved"
            )
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📈 Quick Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Performance by Generation")
            gen_stats = df.groupby('generation')[primary_metric].agg(['mean', 'std', 'max']).round(4)
            gen_stats.columns = ['Mean', 'Std Dev', 'Max']
            st.dataframe(gen_stats, use_container_width=True)
        
        with col2:
            st.markdown("##### Training Time Statistics")
            time_stats = df.groupby('generation')['train_time_sec'].agg(['mean', 'median', 'max']).round(2)
            time_stats.columns = ['Mean (s)', 'Median (s)', 'Max (s)']
            st.dataframe(time_stats, use_container_width=True)
        
        st.markdown("---")
        
        # Best model info
        st.subheader("🏆 Best Model Information")
        best_idx = df[primary_metric].idxmax()
        best_model = df.loc[best_idx]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Model Details")
            st.write(f"**Model:** {best_model['model']}")
            st.write(f"**Generation:** {best_model['generation']}")
            st.write(f"**Training Time:** {best_model['train_time_sec']:.2f}s")
        
        with col2:
            st.markdown("##### Preprocessing Config")
            st.write(f"**Scaler:** {best_model['scaler'] if pd.notna(best_model['scaler']) else 'None'}")
            st.write(f"**Imbalance:** {best_model['imbalance'] if pd.notna(best_model['imbalance']) else 'None'}")
            st.write(f"**Feature Selection:** {best_model['feature_selection'] if pd.notna(best_model['feature_selection']) else 'None'}")
        
        with col3:
            st.markdown("##### Performance Metrics")
            st.write(f"**PR-AUC:** {best_model['pr_auc']:.4f}")
            st.write(f"**Sensitivity:** {best_model['sensitivity']:.4f}")
            st.write(f"**Specificity:** {best_model['specificity']:.4f}")
    
    # Tab 2: Top Models
    with tab2:
        st.header(f"🏆 Top {top_n} Models by {primary_metric.upper().replace('_', '-')}")
        
        top_models = df.nlargest(top_n, primary_metric)
        
        # Metrics comparison
        st.subheader("📊 Metrics Comparison")
        if len(selected_models) > 0:
            fig = create_metrics_comparison(df, selected_models)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one model from the sidebar")
        
        st.markdown("---")
        
        # Radar chart
        st.subheader("🎯 Multi-Metric Radar Chart")
        if len(selected_models) > 0:
            radar_models = selected_models[:5]  # Limit to 5 for clarity
            fig = create_radar_chart(df, radar_models)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Top models table
        st.subheader(f"📋 Top {top_n} Models Table")
        display_cols = ['model', 'generation', 'pr_auc', 'sensitivity', 'specificity', 
                       'f1', 'roc_auc', 'mcc', 'train_time_sec']
        
        styled_df = top_models[display_cols].style.background_gradient(
            subset=['pr_auc', 'sensitivity', 'specificity', 'f1', 'roc_auc', 'mcc'],
            cmap='RdYlGn'
        ).format({
            'pr_auc': '{:.4f}',
            'sensitivity': '{:.4f}',
            'specificity': '{:.4f}',
            'f1': '{:.4f}',
            'roc_auc': '{:.4f}',
            'mcc': '{:.4f}',
            'train_time_sec': '{:.2f}s'
        })
        
        st.dataframe(styled_df, use_container_width=True)
    
    # Tab 3: Generation Analysis
    with tab3:
        st.header("📈 Generation-wise Analysis")
        
        # Generation comparison
        st.subheader("📊 Performance by Generation")
        fig = create_generation_comparison(df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed generation stats
        st.subheader("📋 Detailed Generation Statistics")
        
        gen_detailed = df.groupby('generation').agg({
            'pr_auc': ['mean', 'std', 'min', 'max'],
            'sensitivity': ['mean', 'std'],
            'specificity': ['mean', 'std'],
            'f1': ['mean', 'std'],
            'train_time_sec': ['mean', 'median', 'max']
        }).round(4)
        
        st.dataframe(gen_detailed, use_container_width=True)
        
        st.markdown("---")
        
        # Box plots
        st.subheader("📦 Distribution by Generation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                df,
                x='generation',
                y='pr_auc',
                color='generation',
                title='PR-AUC Distribution',
                labels={'generation': 'Generation', 'pr_auc': 'PR-AUC'}
            )
            fig.update_layout(showlegend=False, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df,
                x='generation',
                y='train_time_sec',
                color='generation',
                title='Training Time Distribution',
                labels={'generation': 'Generation', 'train_time_sec': 'Training Time (s)'}
            )
            fig.update_layout(showlegend=False, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Preprocessing Impact
    with tab4:
        st.header("🔧 Preprocessing Configuration Impact")
        
        # Top preprocessing configs
        st.subheader("🏆 Best Preprocessing Configurations")
        fig = create_preprocessing_impact(df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Individual component analysis
        st.subheader("📊 Component-wise Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Scaler Impact")
            scaler_impact = df.groupby('scaler')[primary_metric].mean().sort_values(ascending=False)
            fig = px.bar(
                x=scaler_impact.index.fillna('None'),
                y=scaler_impact.values,
                labels={'x': 'Scaler', 'y': primary_metric.upper()},
                title=f'Average {primary_metric.upper()} by Scaler',
                text=scaler_impact.values.round(4)
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### Imbalance Handling Impact")
            imb_impact = df.groupby('imbalance')[primary_metric].mean().sort_values(ascending=False)
            fig = px.bar(
                x=imb_impact.index.fillna('None'),
                y=imb_impact.values,
                labels={'x': 'Imbalance Method', 'y': primary_metric.upper()},
                title=f'Average {primary_metric.upper()} by Imbalance Method',
                text=imb_impact.values.round(4),
                color=imb_impact.values,
                color_continuous_scale='Viridis'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("##### Feature Selection Impact")
            fs_impact = df.groupby('feature_selection')[primary_metric].mean().sort_values(ascending=False)
            fig = px.bar(
                x=fs_impact.index.fillna('None'),
                y=fs_impact.values,
                labels={'x': 'Feature Selection', 'y': primary_metric.upper()},
                title=f'Average {primary_metric.upper()} by Feature Selection',
                text=fs_impact.values.round(4),
                color=fs_impact.values,
                color_continuous_scale='Plasma'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Performance Analysis
    with tab5:
        st.header("⏱️ Performance vs Efficiency Analysis")
        
        # Training time vs performance
        st.subheader("📈 Performance vs Training Time Trade-off")
        fig = create_training_time_analysis(df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Efficiency score
        st.subheader("⚡ Efficiency Score (Performance / Time)")
        
        df_efficiency = df.copy()
        df_efficiency['efficiency'] = df_efficiency[primary_metric] / (df_efficiency['train_time_sec'] + 1)
        
        top_efficient = df_efficiency.nlargest(15, 'efficiency')
        
        fig = px.bar(
            top_efficient,
            x='efficiency',
            y='model',
            orientation='h',
            title=f'Top 15 Most Efficient Models ({primary_metric.upper()} / Time)',
            labels={'efficiency': 'Efficiency Score', 'model': 'Model'},
            text='efficiency',
            color='efficiency',
            color_continuous_scale='Greens'
        )
        fig.update_traces(texttemplate='%{text:.6f}', textposition='outside')
        fig.update_layout(height=600, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Metrics correlation
        st.subheader("🔗 Metrics Correlation")
        fig = create_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Scatter matrix
        if len(selected_models) > 0:
            st.subheader("🔍 Metrics Scatter Matrix")
            fig = create_scatter_matrix(df, selected_models[:10])
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 6: Detailed Data
    with tab6:
        st.header("📋 Detailed Experiment Data")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scaler_filter = st.multiselect(
                "Filter by Scaler",
                options=df['scaler'].unique().tolist(),
                default=None
            )
        
        with col2:
            imbalance_filter = st.multiselect(
                "Filter by Imbalance Method",
                options=df['imbalance'].unique().tolist(),
                default=None
            )
        
        with col3:
            fs_filter = st.multiselect(
                "Filter by Feature Selection",
                options=df['feature_selection'].unique().tolist(),
                default=None
            )
        
        # Apply filters
        filtered_data = df.copy()
        
        if scaler_filter:
            filtered_data = filtered_data[filtered_data['scaler'].isin(scaler_filter)]
        if imbalance_filter:
            filtered_data = filtered_data[filtered_data['imbalance'].isin(imbalance_filter)]
        if fs_filter:
            filtered_data = filtered_data[filtered_data['feature_selection'].isin(fs_filter)]
        if selected_gen:
            filtered_data = filtered_data[filtered_data['generation'].isin(selected_gen)]
        
        st.write(f"**Showing {len(filtered_data)} of {len(df)} experiments**")
        
        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            options=['pr_auc', 'sensitivity', 'specificity', 'f1', 'roc_auc', 'mcc', 'train_time_sec'],
            format_func=lambda x: x.upper().replace('_', '-')
        )
        
        sort_order = st.radio("Sort order", ["Descending", "Ascending"], horizontal=True)
        
        sorted_data = filtered_data.sort_values(
            sort_by,
            ascending=(sort_order == "Ascending")
        )
        
        # Display data
        st.dataframe(
            sorted_data.style.background_gradient(
                subset=['pr_auc', 'sensitivity', 'specificity', 'f1', 'roc_auc', 'mcc'],
                cmap='RdYlGn'
            ).format({
                'pr_auc': '{:.4f}',
                'pr_auc_std': '{:.4f}',
                'sensitivity': '{:.4f}',
                'specificity': '{:.4f}',
                'f1': '{:.4f}',
                'roc_auc': '{:.4f}',
                'mcc': '{:.4f}',
                'npv': '{:.4f}',
                'precision': '{:.4f}',
                'train_time_sec': '{:.2f}s'
            }),
            use_container_width=True,
            height=600
        )
        
        # Download button
        csv = sorted_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🔬 Classification Models Comparison Dashboard | 
            Built with Streamlit & Plotly | 
            Data: {}</p>
        </div>
        """.format(filename),
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
