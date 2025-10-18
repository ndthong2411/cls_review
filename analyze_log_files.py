"""
Script phân tích và mô tả file log từ experiments
Phân tích kết quả thực nghiệm trên cardio_train và creditcard datasets
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

class LogAnalyzer:
    def __init__(self, log_file_path):
        self.log_path = Path(log_file_path)
        self.log_content = self.log_path.read_text(encoding='utf-8')
        self.dataset_name = None
        self.metadata = {}
        self.experiments = []

    def parse_metadata(self):
        """Phân tích metadata từ log"""
        # Dataset name
        dataset_match = re.search(r'Dataset: (\w+)', self.log_content)
        if dataset_match:
            self.dataset_name = dataset_match.group(1)

        # Start time
        time_match = re.search(r'Start time: ([\d\-\s:]+)', self.log_content)
        if time_match:
            self.metadata['start_time'] = time_match.group(1)

        # Dataset info
        shape_match = re.search(r'Original shape: \((\d+), (\d+)\)', self.log_content)
        if shape_match:
            self.metadata['rows'] = int(shape_match.group(1))
            self.metadata['cols'] = int(shape_match.group(2))

        # Features
        features_match = re.search(r'Features: (\d+)', self.log_content)
        if features_match:
            self.metadata['features'] = int(features_match.group(1))

        # Class distribution
        pos_match = re.search(r'Positive class: (\d+) \(([\d.]+)%\)', self.log_content)
        neg_match = re.search(r'Negative class: (\d+) \(([\d.]+)%\)', self.log_content)
        if pos_match and neg_match:
            self.metadata['positive'] = int(pos_match.group(1))
            self.metadata['positive_pct'] = float(pos_match.group(2))
            self.metadata['negative'] = int(neg_match.group(1))
            self.metadata['negative_pct'] = float(neg_match.group(2))

        # Train/test split
        train_match = re.search(r'Train: (\d+) samples \((\d+) positive\)', self.log_content)
        test_match = re.search(r'Test:\s+(\d+) samples \((\d+) positive\)', self.log_content)
        if train_match and test_match:
            self.metadata['train_samples'] = int(train_match.group(1))
            self.metadata['train_positive'] = int(train_match.group(2))
            self.metadata['test_samples'] = int(test_match.group(1))
            self.metadata['test_positive'] = int(test_match.group(2))

        # Total experiments
        exp_match = re.search(r'Total experiments: (\d+)', self.log_content)
        if exp_match:
            self.metadata['total_experiments'] = int(exp_match.group(1))

        # Cache info
        cache_match = re.search(r'Cached experiments: (\d+)', self.log_content)
        if cache_match:
            self.metadata['cached_experiments'] = int(cache_match.group(1))

    def parse_experiments(self):
        """Phân tích kết quả từng experiment"""
        # Pattern: [1/270] Gen1_LogisticRegression | Scale: standard | Imb: none | FeatSel: none
        #   [CACHE] Loaded from cache!
        #   [OK] PR-AUC: 0.7580 | Sens: 0.6525 | Spec: 0.7960 | F1: 0.7028 | Time: 0.2s

        pattern = r'\[(\d+)/(\d+)\] ([^\|]+) \| Scale: ([^\|]+) \| Imb: ([^\|]+) \| FeatSel: ([^\n]+)\n.*?\n.*?\[(?:OK|CACHE|ERROR)\] PR-AUC: ([\d.]+) \| Sens: ([\d.]+) \| Spec: ([\d.]+) \| F1: ([\d.]+) \| Time: ([\d.]+)s'

        matches = re.findall(pattern, self.log_content, re.DOTALL)

        for match in matches:
            exp = {
                'exp_num': int(match[0]),
                'total': int(match[1]),
                'model': match[2].strip(),
                'scaler': match[3].strip(),
                'imbalance': match[4].strip(),
                'feature_selection': match[5].strip(),
                'pr_auc': float(match[6]),
                'sensitivity': float(match[7]),
                'specificity': float(match[8]),
                'f1_score': float(match[9]),
                'time': float(match[10])
            }
            self.experiments.append(exp)

    def get_summary(self):
        """Tạo summary báo cáo"""
        summary = {
            'dataset': self.dataset_name,
            'metadata': self.metadata,
            'experiments_parsed': len(self.experiments)
        }

        if self.experiments:
            df = pd.DataFrame(self.experiments)

            # Top models by PR-AUC
            summary['top_5_pr_auc'] = df.nlargest(5, 'pr_auc')[['model', 'scaler', 'imbalance', 'feature_selection', 'pr_auc', 'f1_score']].to_dict('records')

            # Best by model generation
            summary['best_by_generation'] = {}
            for gen in ['Gen1', 'Gen2', 'Gen3', 'Gen4']:
                gen_df = df[df['model'].str.startswith(gen)]
                if len(gen_df) > 0:
                    best = gen_df.loc[gen_df['pr_auc'].idxmax()]
                    summary['best_by_generation'][gen] = {
                        'model': best['model'],
                        'pr_auc': best['pr_auc'],
                        'f1_score': best['f1_score'],
                        'config': f"{best['scaler']} | {best['imbalance']} | {best['feature_selection']}"
                    }

            # Statistics
            summary['stats'] = {
                'avg_pr_auc': df['pr_auc'].mean(),
                'max_pr_auc': df['pr_auc'].max(),
                'min_pr_auc': df['pr_auc'].min(),
                'avg_f1': df['f1_score'].mean(),
                'avg_time': df['time'].mean(),
                'total_time': df['time'].sum()
            }

        return summary

    def create_visualizations(self, output_dir='analysis_output'):
        """Tạo biểu đồ phân tích"""
        if not self.experiments:
            print("No experiments to visualize")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.experiments)

        # 1. PR-AUC by model
        plt.figure(figsize=(14, 6))
        model_avg = df.groupby('model')['pr_auc'].mean().sort_values(ascending=False)
        plt.bar(range(len(model_avg)), model_avg.values)
        plt.xticks(range(len(model_avg)), model_avg.index, rotation=45, ha='right')
        plt.ylabel('Average PR-AUC')
        plt.title(f'{self.dataset_name}: Average PR-AUC by Model')
        plt.tight_layout()
        plt.savefig(output_path / f'{self.dataset_name}_pr_auc_by_model.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Impact of preprocessing techniques
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Scaler impact
        scaler_avg = df.groupby('scaler')['pr_auc'].mean().sort_values(ascending=False)
        axes[0].bar(range(len(scaler_avg)), scaler_avg.values)
        axes[0].set_xticks(range(len(scaler_avg)))
        axes[0].set_xticklabels(scaler_avg.index, rotation=45, ha='right')
        axes[0].set_ylabel('Average PR-AUC')
        axes[0].set_title('Impact of Scaler')

        # Imbalance handling impact
        imb_avg = df.groupby('imbalance')['pr_auc'].mean().sort_values(ascending=False)
        axes[1].bar(range(len(imb_avg)), imb_avg.values)
        axes[1].set_xticks(range(len(imb_avg)))
        axes[1].set_xticklabels(imb_avg.index, rotation=45, ha='right')
        axes[1].set_ylabel('Average PR-AUC')
        axes[1].set_title('Impact of Imbalance Handling')

        # Feature selection impact
        feat_avg = df.groupby('feature_selection')['pr_auc'].mean().sort_values(ascending=False)
        axes[2].bar(range(len(feat_avg)), feat_avg.values)
        axes[2].set_xticks(range(len(feat_avg)))
        axes[2].set_xticklabels(feat_avg.index, rotation=45, ha='right')
        axes[2].set_ylabel('Average PR-AUC')
        axes[2].set_title('Impact of Feature Selection')

        plt.tight_layout()
        plt.savefig(output_path / f'{self.dataset_name}_preprocessing_impact.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. PR-AUC vs F1 Score scatter
        plt.figure(figsize=(10, 8))
        for gen in ['Gen1', 'Gen2', 'Gen3', 'Gen4']:
            gen_df = df[df['model'].str.startswith(gen)]
            plt.scatter(gen_df['pr_auc'], gen_df['f1_score'], label=gen, alpha=0.6, s=50)
        plt.xlabel('PR-AUC')
        plt.ylabel('F1 Score')
        plt.title(f'{self.dataset_name}: PR-AUC vs F1 Score by Generation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f'{self.dataset_name}_pr_auc_vs_f1.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Heatmap: Model vs Preprocessing
        pivot = df.groupby(['model', 'scaler'])['pr_auc'].mean().unstack()
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'PR-AUC'})
        plt.title(f'{self.dataset_name}: PR-AUC Heatmap (Model × Scaler)')
        plt.tight_layout()
        plt.savefig(output_path / f'{self.dataset_name}_heatmap_model_scaler.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Saved visualizations to {output_path}")

    def print_report(self):
        """In báo cáo chi tiết"""
        summary = self.get_summary()

        print("=" * 80)
        print(f"PHÂN TÍCH LOG: {self.dataset_name.upper()}")
        print("=" * 80)

        print("\n[METADATA]")
        print(f"  - Dataset: {self.dataset_name}")
        print(f"  - Start time: {self.metadata.get('start_time', 'N/A')}")
        print(f"  - Original shape: {self.metadata.get('rows', 'N/A')} rows x {self.metadata.get('cols', 'N/A')} cols")
        print(f"  - Features used: {self.metadata.get('features', 'N/A')}")

        print("\n[CLASS DISTRIBUTION]")
        print(f"  - Positive: {self.metadata.get('positive', 'N/A')} ({self.metadata.get('positive_pct', 'N/A')}%)")
        print(f"  - Negative: {self.metadata.get('negative', 'N/A')} ({self.metadata.get('negative_pct', 'N/A')}%)")
        print(f"  - Imbalance ratio: 1:{self.metadata.get('negative', 0) / max(self.metadata.get('positive', 1), 1):.1f}")

        print("\n[TRAIN/TEST SPLIT]")
        print(f"  - Train: {self.metadata.get('train_samples', 'N/A')} ({self.metadata.get('train_positive', 'N/A')} positive)")
        print(f"  - Test: {self.metadata.get('test_samples', 'N/A')} ({self.metadata.get('test_positive', 'N/A')} positive)")

        print("\n[EXPERIMENTS]")
        print(f"  - Total planned: {self.metadata.get('total_experiments', 'N/A')}")
        print(f"  - Cached: {self.metadata.get('cached_experiments', 'N/A')}")
        print(f"  - Parsed successfully: {summary['experiments_parsed']}")

        if 'stats' in summary:
            print("\n[STATISTICS]")
            print(f"  - Average PR-AUC: {summary['stats']['avg_pr_auc']:.4f}")
            print(f"  - Best PR-AUC: {summary['stats']['max_pr_auc']:.4f}")
            print(f"  - Worst PR-AUC: {summary['stats']['min_pr_auc']:.4f}")
            print(f"  - Average F1: {summary['stats']['avg_f1']:.4f}")
            print(f"  - Average time/exp: {summary['stats']['avg_time']:.2f}s")
            print(f"  - Total time: {summary['stats']['total_time']:.2f}s ({summary['stats']['total_time']/60:.1f} min)")

        if 'top_5_pr_auc' in summary:
            print("\n[TOP 5 EXPERIMENTS - by PR-AUC]")
            for i, exp in enumerate(summary['top_5_pr_auc'], 1):
                print(f"\n  {i}. {exp['model']}")
                print(f"     Config: {exp['scaler']} | {exp['imbalance']} | {exp['feature_selection']}")
                print(f"     PR-AUC: {exp['pr_auc']:.4f} | F1: {exp['f1_score']:.4f}")

        if 'best_by_generation' in summary:
            print("\n[BEST MODEL PER GENERATION]")
            for gen in ['Gen1', 'Gen2', 'Gen3', 'Gen4']:
                if gen in summary['best_by_generation']:
                    best = summary['best_by_generation'][gen]
                    print(f"\n  {gen}: {best['model']}")
                    print(f"     PR-AUC: {best['pr_auc']:.4f} | F1: {best['f1_score']:.4f}")
                    print(f"     Config: {best['config']}")

        print("\n" + "=" * 80 + "\n")


def main():
    """Phân tích cả 2 log files"""
    log_files = [
        r"E:\thong\code\cls_review\datathucte\cardio_train_20251018_022847.log",
        r"E:\thong\code\cls_review\datathucte\creditcard_20251018_204737.log"
    ]

    analyzers = []

    for log_file in log_files:
        if not Path(log_file).exists():
            print(f"[X] File not found: {log_file}")
            continue

        print(f"[*] Analyzing: {log_file}")
        analyzer = LogAnalyzer(log_file)
        analyzer.parse_metadata()
        analyzer.parse_experiments()
        analyzer.print_report()
        analyzers.append(analyzer)

    # Tạo visualizations
    print("[*] Generating visualizations...")
    for analyzer in analyzers:
        analyzer.create_visualizations(output_dir=f'analysis_output/{analyzer.dataset_name}')

    # So sánh 2 datasets
    if len(analyzers) == 2:
        print("\n" + "=" * 80)
        print("SO SÁNH 2 DATASETS")
        print("=" * 80)

        for analyzer in analyzers:
            summary = analyzer.get_summary()
            print(f"\n{analyzer.dataset_name.upper()}:")
            print(f"  • Class imbalance: {analyzer.metadata.get('negative', 0) / max(analyzer.metadata.get('positive', 1), 1):.1f}:1")
            if 'stats' in summary:
                print(f"  • Best PR-AUC: {summary['stats']['max_pr_auc']:.4f}")
                print(f"  • Avg PR-AUC: {summary['stats']['avg_pr_auc']:.4f}")
                print(f"  • Total time: {summary['stats']['total_time']/60:.1f} minutes")

    print("\n[OK] Analysis complete!")


if __name__ == "__main__":
    main()
