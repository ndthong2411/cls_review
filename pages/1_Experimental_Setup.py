"""
🔬 Experimental Setup Page - Detailed overview of datasets, models, and configurations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Experimental Setup",
    page_icon="🔬",
    layout="wide"
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
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .dataset-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 1rem;
    }
    .cardio-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .credit-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .gen-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .gen1 { background-color: #e3f2fd; color: #1976d2; }
    .gen2 { background-color: #f3e5f5; color: #7b1fa2; }
    .gen3 { background-color: #e8f5e9; color: #388e3c; }
    .gen4 { background-color: #fff3e0; color: #f57c00; }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_dataset_info():
    """Load and analyze raw datasets"""
    info = {}

    # Cardio dataset
    try:
        cardio = pd.read_csv('data/raw/cardio_train.csv', sep=';')
        cardio_target = cardio['cardio']
        info['cardio'] = {
            'total_samples': len(cardio),
            'n_features': len(cardio.columns) - 1,
            'feature_names': [col for col in cardio.columns if col != 'cardio'],
            'class_0': (cardio_target == 0).sum(),
            'class_1': (cardio_target == 1).sum(),
            'imbalance_ratio': (cardio_target == 0).sum() / (cardio_target == 1).sum(),
            'dataset': cardio
        }
    except:
        info['cardio'] = None

    # CreditCard dataset
    try:
        credit = pd.read_csv('data/raw/creditcard.csv')
        credit_target = credit['Class']
        info['credit'] = {
            'total_samples': len(credit),
            'n_features': len(credit.columns) - 1,
            'feature_names': [col for col in credit.columns if col != 'Class'],
            'class_0': (credit_target == 0).sum(),
            'class_1': (credit_target == 1).sum(),
            'imbalance_ratio': (credit_target == 0).sum() / (credit_target == 1).sum(),
            'dataset': credit
        }
    except:
        info['credit'] = None

    return info

@st.cache_data
def simulate_imbalance_impact():
    """Simulate SMOTE/SMOTEENN impact on datasets"""
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN

    results = {}

    # Cardio
    try:
        cardio = pd.read_csv('data/raw/cardio_train.csv', sep=';')
        feature_cols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                       'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        X = cardio[feature_cols]
        y = cardio['cardio']

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Original
        original_counts = y_train.value_counts().sort_index()

        # SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        smote_counts = pd.Series(y_smote).value_counts().sort_index()

        # SMOTEENN
        smoteenn = SMOTEENN(random_state=42)
        X_enn, y_enn = smoteenn.fit_resample(X_train, y_train)
        enn_counts = pd.Series(y_enn).value_counts().sort_index()

        results['cardio'] = {
            'original': {'total': len(y_train), 'class_0': original_counts[0], 'class_1': original_counts[1]},
            'smote': {'total': len(y_smote), 'class_0': smote_counts[0], 'class_1': smote_counts[1]},
            'smoteenn': {'total': len(y_enn), 'class_0': enn_counts[0], 'class_1': enn_counts[1]}
        }
    except Exception as e:
        st.warning(f"Could not simulate cardio: {e}")
        results['cardio'] = None

    # CreditCard
    try:
        credit = pd.read_csv('data/raw/creditcard.csv')
        X = credit.drop('Class', axis=1)
        y = credit['Class']

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        original_counts = y_train.value_counts().sort_index()

        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        smote_counts = pd.Series(y_smote).value_counts().sort_index()

        smoteenn = SMOTEENN(random_state=42)
        X_enn, y_enn = smoteenn.fit_resample(X_train, y_train)
        enn_counts = pd.Series(y_enn).value_counts().sort_index()

        results['credit'] = {
            'original': {'total': len(y_train), 'class_0': original_counts[0], 'class_1': original_counts[1]},
            'smote': {'total': len(y_smote), 'class_0': smote_counts[0], 'class_1': smote_counts[1]},
            'smoteenn': {'total': len(y_enn), 'class_0': enn_counts[0], 'class_1': enn_counts[1]}
        }
    except Exception as e:
        st.warning(f"Could not simulate credit: {e}")
        results['credit'] = None

    return results

# Main content
def main():
    st.markdown('<h1 class="main-header">🔬 Thiết Lập Thực Nghiệm</h1>', unsafe_allow_html=True)

    st.markdown("""
    Trang này giới thiệu chi tiết về **datasets**, **models**, **preprocessing configurations**,
    và **ảnh hưởng thực tế** của các phương pháp xử lý dữ liệu.
    """)

    # ==================== SECTION 1: DATASETS ====================
    st.markdown('<h2 class="section-header">📊 Datasets Overview</h2>', unsafe_allow_html=True)

    dataset_info = load_dataset_info()

    col1, col2 = st.columns(2)

    # Cardio Dataset
    with col1:
        st.markdown("""
        <div class="dataset-card cardio-card">
            <h3>❤️ Cardio Train Dataset</h3>
            <p><strong>Medical diagnosis - Cardiovascular disease detection</strong></p>
        </div>
        """, unsafe_allow_html=True)

        if dataset_info['cardio']:
            info = dataset_info['cardio']

            st.metric("📦 Total Samples", f"{info['total_samples']:,}")
            st.metric("📏 Features", f"{info['n_features']}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("✅ Class 0 (Healthy)", f"{info['class_0']:,}",
                         delta=f"{info['class_0']/info['total_samples']*100:.1f}%")
            with col_b:
                st.metric("❌ Class 1 (Disease)", f"{info['class_1']:,}",
                         delta=f"{info['class_1']/info['total_samples']*100:.1f}%")

            st.metric("⚖️ Imbalance Ratio", f"{info['imbalance_ratio']:.2f}:1")

            # Class distribution
            fig = go.Figure(data=[
                go.Bar(name='Class 0', x=['Healthy'], y=[info['class_0']], marker_color='#2ecc71'),
                go.Bar(name='Class 1', x=['Disease'], y=[info['class_1']], marker_color='#e74c3c')
            ])
            fig.update_layout(
                title='Class Distribution',
                yaxis_title='Number of Samples',
                height=300,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("🔍 Feature Details"):
                st.write("**Original Features:**")
                st.code(", ".join(info['feature_names'][:11]), language=None)

                st.write("**Engineered Features (added in preprocessing):**")
                st.code("age_years, bmi, pulse_pressure, map, is_hypertension", language=None)

                st.info("""
                **Total features used:** 15
                - 11 original features
                - 4 engineered features
                """)
        else:
            st.error("Cardio dataset not found!")

    # CreditCard Dataset
    with col2:
        st.markdown("""
        <div class="dataset-card credit-card">
            <h3>💳 Credit Card Dataset</h3>
            <p><strong>Fraud detection - Highly imbalanced binary classification</strong></p>
        </div>
        """, unsafe_allow_html=True)

        if dataset_info['credit']:
            info = dataset_info['credit']

            st.metric("📦 Total Samples", f"{info['total_samples']:,}")
            st.metric("📏 Features", f"{info['n_features']}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("✅ Class 0 (Normal)", f"{info['class_0']:,}",
                         delta=f"{info['class_0']/info['total_samples']*100:.2f}%")
            with col_b:
                st.metric("🚨 Class 1 (Fraud)", f"{info['class_1']:,}",
                         delta=f"{info['class_1']/info['total_samples']*100:.2f}%",
                         delta_color="inverse")

            st.metric("⚖️ Imbalance Ratio", f"{info['imbalance_ratio']:.0f}:1")
            st.error(f"⚠️ SEVERELY IMBALANCED! Only {info['class_1']/info['total_samples']*100:.3f}% are fraud cases")

            # Class distribution
            fig = go.Figure(data=[
                go.Bar(name='Class 0', x=['Normal'], y=[info['class_0']], marker_color='#3498db'),
                go.Bar(name='Class 1', x=['Fraud'], y=[info['class_1']], marker_color='#e67e22')
            ])
            fig.update_layout(
                title='Class Distribution (Log Scale)',
                yaxis_title='Number of Samples (log scale)',
                yaxis_type='log',
                height=300,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("🔍 Feature Details"):
                st.write("**Features:** 30 features")
                st.info("""
                - **Time**: Seconds since first transaction
                - **Amount**: Transaction amount
                - **V1-V28**: PCA-transformed features (anonymized for privacy)

                All features are numerical, already preprocessed by PCA.
                """)
        else:
            st.error("CreditCard dataset not found!")

    # ==================== SECTION 2: MODELS ====================
    st.markdown('<h2 class="section-header">🤖 Models by Generation</h2>', unsafe_allow_html=True)

    models_info = {
        'Generation 1': {
            'badge_class': 'gen1',
            'title': 'Baseline (Classical ML)',
            'models': [
                ('LogisticRegression', 'Linear classifier', '✅', 'Fast'),
                ('DecisionTree', 'Tree-based', '❌', 'Fast'),
                ('KNN', 'Distance-based', '✅', 'Slow (predict)')
            ],
            'description': 'Simple, interpretable models for baseline comparison'
        },
        'Generation 2': {
            'badge_class': 'gen2',
            'title': 'Intermediate (Ensemble Learning)',
            'models': [
                ('RandomForest', 'Bagging ensemble', '❌', 'Fast'),
                ('ExtraTrees', 'Extra randomization', '❌', 'Fast'),
                ('GradientBoosting', 'Sequential boosting', '❌', 'Medium'),
                ('SVM_RBF', 'Kernel method', '✅', 'Very Slow'),
                ('MLP_Sklearn', 'Neural network', '✅', 'Medium')
            ],
            'description': 'More powerful ensemble methods and neural networks'
        },
        'Generation 3': {
            'badge_class': 'gen3',
            'title': 'Advanced (Gradient Boosting SOTA)',
            'models': [
                ('XGBoost', 'eXtreme GB + GPU', '❌', 'Fast (GPU)'),
                ('LightGBM', 'Light GB + GPU', '❌', 'Very Fast (GPU)'),
                ('CatBoost', 'Categorical GB + GPU', '❌', 'Slow')
            ],
            'description': 'State-of-the-art boosting with GPU acceleration'
        },
        'Generation 4': {
            'badge_class': 'gen4',
            'title': 'Deep Learning (SOTA)',
            'models': [
                ('PyTorch_MLP', 'Deep neural network', '✅', 'Medium (GPU)'),
                ('TabNet', 'Attention-based', '❌', 'Medium (GPU)')
            ],
            'description': 'Deep learning architectures for tabular data'
        }
    }

    for gen, info in models_info.items():
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <span class="gen-badge {info['badge_class']}">{gen}</span>
            <strong style="margin-left: 1rem;">{info['title']}</strong>
        </div>
        """, unsafe_allow_html=True)

        st.caption(info['description'])

        # Models table
        df_models = pd.DataFrame(info['models'], columns=['Model', 'Type', 'Needs Scaling', 'Speed'])
        st.dataframe(df_models, use_container_width=True, hide_index=True)

        st.markdown("---")

    # Summary
    total_models = sum(len(info['models']) for info in models_info.values())
    st.info(f"""
    📊 **Total Models:** {total_models} different architectures across 4 generations

    🚀 **GPU Support:** Generation 3 & 4 models utilize GPU acceleration

    ⚖️ **Scaling Requirements:**
    - Distance-based & Neural networks: REQUIRE scaling
    - Tree-based models: DON'T need scaling
    """)

    # ==================== SECTION 3: PREPROCESSING CONFIGS ====================
    st.markdown('<h2 class="section-header">🔧 Preprocessing Configurations</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📐 Scaling Methods")
        st.markdown("""
        **1. None** - No scaling (for tree-based models)

        **2. Standard Scaler**
        ```python
        X_scaled = (X - mean) / std
        ```
        ✅ Good for: Normally distributed data

        **3. Robust Scaler**
        ```python
        X_scaled = (X - median) / IQR
        ```
        ✅ Good for: Data with outliers
        """)

        st.info("**Total combinations:** 3 scaling methods")

    with col2:
        st.markdown("### 🎯 Feature Selection")
        st.markdown("""
        **1. None** - Use all features

        **2. SelectKBest (k=5)** - Top 5 features by F-score

        **3. SelectKBest (k=12)** - Top 12 features by F-score

        **4. MutualInfo (k=5)** - Top 5 by mutual information

        **5. MutualInfo (k=12)** - Top 12 by mutual information
        """)

        st.info("**Total combinations:** 5 feature selection methods")

    with col3:
        st.markdown("### ⚖️ Imbalance Handling")
        st.markdown("""
        **1. None** - No sampling

        **2. SMOTE**
        - Synthetic Minority Over-sampling
        - Creates synthetic samples for minority class

        **3. SMOTE-ENN**
        - SMOTE + Edited Nearest Neighbors
        - Oversample then remove noisy samples
        """)

        st.info("**Total combinations:** 3 imbalance methods")

    # Total experiments
    st.success(f"""
    🎯 **Total Experimental Configurations:**
    - Scaling: 3 methods
    - Feature Selection: 5 methods
    - Imbalance: 3 methods
    - Models: {total_models} architectures

    **Total experiments:** 3 × 5 × 3 = 45 configs per model = **{45 * total_models} total experiments**

    *(Actual number may vary based on model-specific requirements)*
    """)

    # ==================== SECTION 4: IMBALANCE IMPACT ====================
    st.markdown('<h2 class="section-header">📈 Ảnh Hưởng Thực Tế của Imbalance Handling</h2>', unsafe_allow_html=True)

    st.markdown("""
    Phần này cho thấy **ảnh hưởng thực tế** của các phương pháp xử lý imbalance (SMOTE, SMOTE-ENN)
    lên **số lượng mẫu** trong train set của 2 datasets.
    """)

    with st.spinner("Đang tính toán ảnh hưởng của imbalance handling..."):
        imbalance_results = simulate_imbalance_impact()

    col1, col2 = st.columns(2)

    # Cardio results
    with col1:
        st.markdown("### ❤️ Cardio Train Dataset")

        if imbalance_results['cardio']:
            results = imbalance_results['cardio']

            # Create comparison table
            comparison_df = pd.DataFrame({
                'Method': ['Original', 'SMOTE', 'SMOTE-ENN'],
                'Total': [results['original']['total'], results['smote']['total'], results['smoteenn']['total']],
                'Class 0': [results['original']['class_0'], results['smote']['class_0'], results['smoteenn']['class_0']],
                'Class 1': [results['original']['class_1'], results['smote']['class_1'], results['smoteenn']['class_1']],
            })

            comparison_df['Change'] = comparison_df['Total'] - results['original']['total']
            comparison_df['Change %'] = (comparison_df['Total'] / results['original']['total'] - 1) * 100

            st.dataframe(comparison_df.style.format({
                'Total': '{:,}',
                'Class 0': '{:,}',
                'Class 1': '{:,}',
                'Change': '{:+,}',
                'Change %': '{:+.1f}%'
            }), use_container_width=True, hide_index=True)

            # Visualization
            fig = go.Figure()

            methods = ['Original', 'SMOTE', 'SMOTE-ENN']
            class_0 = [results['original']['class_0'], results['smote']['class_0'], results['smoteenn']['class_0']]
            class_1 = [results['original']['class_1'], results['smote']['class_1'], results['smoteenn']['class_1']]

            fig.add_trace(go.Bar(name='Class 0', x=methods, y=class_0, marker_color='#2ecc71'))
            fig.add_trace(go.Bar(name='Class 1', x=methods, y=class_1, marker_color='#e74c3c'))

            fig.update_layout(
                title='Sample Distribution by Method',
                yaxis_title='Number of Samples',
                barmode='group',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Key insights
            st.info(f"""
            📊 **Key Insights:**
            - Original: {results['original']['total']:,} samples (50/50 balanced)
            - SMOTE: {results['smote']['total']:,} samples ({results['smote']['total'] - results['original']['total']:+,} change)
            - SMOTE-ENN: {results['smoteenn']['total']:,} samples ({results['smoteenn']['total'] - results['original']['total']:+,} change)

            ⚠️ **Note:** SMOTE-ENN removes noisy samples → Can DECREASE total samples!
            """)

    # CreditCard results
    with col2:
        st.markdown("### 💳 CreditCard Train Dataset")

        if imbalance_results['credit']:
            results = imbalance_results['credit']

            # Create comparison table
            comparison_df = pd.DataFrame({
                'Method': ['Original', 'SMOTE', 'SMOTE-ENN'],
                'Total': [results['original']['total'], results['smote']['total'], results['smoteenn']['total']],
                'Class 0': [results['original']['class_0'], results['smote']['class_0'], results['smoteenn']['class_0']],
                'Class 1': [results['original']['class_1'], results['smote']['class_1'], results['smoteenn']['class_1']],
            })

            comparison_df['Change'] = comparison_df['Total'] - results['original']['total']
            comparison_df['Change %'] = (comparison_df['Total'] / results['original']['total'] - 1) * 100

            st.dataframe(comparison_df.style.format({
                'Total': '{:,}',
                'Class 0': '{:,}',
                'Class 1': '{:,}',
                'Change': '{:+,}',
                'Change %': '{:+.1f}%'
            }), use_container_width=True, hide_index=True)

            # Visualization
            fig = go.Figure()

            methods = ['Original', 'SMOTE', 'SMOTE-ENN']
            class_0 = [results['original']['class_0'], results['smote']['class_0'], results['smoteenn']['class_0']]
            class_1 = [results['original']['class_1'], results['smote']['class_1'], results['smoteenn']['class_1']]

            fig.add_trace(go.Bar(name='Class 0', x=methods, y=class_0, marker_color='#3498db'))
            fig.add_trace(go.Bar(name='Class 1', x=methods, y=class_1, marker_color='#e67e22'))

            fig.update_layout(
                title='Sample Distribution by Method',
                yaxis_title='Number of Samples',
                barmode='group',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Key insights
            fraud_increase = results['smote']['class_1'] - results['original']['class_1']
            st.warning(f"""
            🚨 **Key Insights:**
            - Original: Only {results['original']['class_1']:,} fraud samples ({results['original']['class_1']/results['original']['total']*100:.2f}%)
            - SMOTE: Creates {fraud_increase:,} synthetic fraud samples!
            - Total increase: {results['smote']['total'] - results['original']['total']:,} samples ({(results['smote']['total']/results['original']['total'] - 1)*100:.1f}%)

            ⚠️ **Critical:** SMOTE nearly DOUBLES the dataset size for severely imbalanced data!
            """)

    # Comparison visualization
    st.markdown("### 📊 So Sánh Ảnh Hưởng Giữa 2 Datasets")

    if imbalance_results['cardio'] and imbalance_results['credit']:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cardio: Change in Total Samples', 'CreditCard: Change in Total Samples'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )

        # Cardio
        cardio_results = imbalance_results['cardio']
        cardio_changes = [
            0,
            cardio_results['smote']['total'] - cardio_results['original']['total'],
            cardio_results['smoteenn']['total'] - cardio_results['original']['total']
        ]

        fig.add_trace(
            go.Bar(
                x=['Original', 'SMOTE', 'SMOTE-ENN'],
                y=cardio_changes,
                marker_color=['gray', '#2ecc71', '#e74c3c'],
                text=[f"{x:+,}" for x in cardio_changes],
                textposition='outside'
            ),
            row=1, col=1
        )

        # CreditCard
        credit_results = imbalance_results['credit']
        credit_changes = [
            0,
            credit_results['smote']['total'] - credit_results['original']['total'],
            credit_results['smoteenn']['total'] - credit_results['original']['total']
        ]

        fig.add_trace(
            go.Bar(
                x=['Original', 'SMOTE', 'SMOTE-ENN'],
                y=credit_changes,
                marker_color=['gray', '#3498db', '#e67e22'],
                text=[f"{x:+,}" for x in credit_changes],
                textposition='outside'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            template='plotly_white',
            showlegend=False
        )
        fig.update_yaxes(title_text="Change in Samples", row=1, col=1)
        fig.update_yaxes(title_text="Change in Samples", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

    # Summary
    st.success("""
    🎯 **Kết Luận:**

    1. **Cardio (Balanced dataset):**
       - SMOTE: Tăng nhẹ để cân bằng hoàn hảo 50/50
       - SMOTE-ENN: GIẢM samples do loại bỏ noise (~67% reduction)
       - **Recommendation:** Use `none` hoặc `smote` (SMOTE-ENN mất quá nhiều data)

    2. **CreditCard (Severely Imbalanced):**
       - SMOTE: Tăng MẠNH ~100% (tạo hàng trăm ngàn synthetic fraud samples)
       - SMOTE-ENN: Tăng ~88% (tạo samples + loại noise)
       - **Recommendation:** Test cả 3 methods, nhưng `none` có thể tốt hơn với tree-based models

    3. **General Rule:**
       - Balanced data (ratio < 2:1): SMOTE ít ảnh hưởng
       - Imbalanced data (ratio > 10:1): SMOTE tăng dataset đáng kể
       - SMOTE-ENN: Có thể giảm samples nếu data nhiều noise
    """)

    # Footer
    st.markdown("---")
    st.caption("🔬 Experimental Setup | 📊 2 Datasets | 🤖 13 Models | 🔧 45 Configs/Model")

if __name__ == "__main__":
    main()
