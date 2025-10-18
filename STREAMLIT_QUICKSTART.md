# 🚀 Streamlit Dashboard - Quick Start

## Chạy Dashboard Ngay (3 Bước)

### Bước 1: Kiểm tra yêu cầu

```bash
# Kiểm tra Streamlit đã cài chưa
pip list | grep streamlit

# Nếu chưa có, cài ngay
pip install streamlit plotly
```

### Bước 2: Chạy dashboard

```bash
streamlit run app_streamlit.py
```

### Bước 3: Mở browser

Dashboard tự động mở tại: **http://localhost:8501**

Nếu không tự mở, copy link vào browser.

---

## Giao Diện

```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   🔬 ML Models Comparison Dashboard                       ║
║                                                            ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Sidebar:                Main Content:                    ║
║  ┌──────────────┐       ┌──────────────────────────┐     ║
║  │ 📁 Dataset   │       │ ✅ Loaded: cardio_train │     ║
║  │ ○ cardio_train│      │                          │     ║
║  │ ○ creditcard │       │ 📊 Overview              │     ║
║  │              │       │ ├─ Total: 270 exps       │     ║
║  │ 📊 Models    │       │ ├─ Best PR-AUC: 0.8023   │     ║
║  │ ☑ Gen1       │       │ └─ Best: DecisionTree    │     ║
║  │ ☑ Gen2       │       │                          │     ║
║  │ ☑ Gen3       │       │ [Tabs: Overview | Models │     ║
║  │ ☑ Gen4       │       │  | Top | Preprocessing]  │     ║
║  │              │       │                          │     ║
║  │ ⚙️ Filters   │       │ 📊 Interactive Charts    │     ║
║  │ • Scaler     │       │                          │     ║
║  │ • Imbalance  │       │                          │     ║
║  │ • Features   │       │                          │     ║
║  └──────────────┘       └──────────────────────────┘     ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

---

## Tính Năng Chính

### 1. **Dataset Selection** (Sidebar)
Chọn dataset để phân tích:
- 🫀 **cardio_train**: Cardiovascular disease (balanced)
- 💳 **creditcard**: Fraud detection (imbalanced)

### 2. **Model Filtering** (Sidebar)
Filter theo:
- ✅ Generation (Gen1, Gen2, Gen3, Gen4)
- ✅ Specific models
- ✅ Preprocessing (scaler, imbalance, features)

### 3. **Interactive Charts**
- 📊 Bar charts - Model comparison
- 📈 Line charts - Performance trends
- 🎯 Scatter plots - PR-AUC vs F1
- 🔥 Heatmaps - Metric correlations

### 4. **Tabs**

#### Tab 1: **Overview**
- Tổng quan metrics
- Key statistics

#### Tab 2: **Model Comparison**
- So sánh performance
- Filter theo generation

#### Tab 3: **Top Performers**
- Top 10 configurations
- Best per generation

#### Tab 4: **Preprocessing Impact**
- Scaler comparison
- Imbalance handling effectiveness
- Feature selection impact

#### Tab 5: **Detailed Results**
- Full table (searchable)
- Export CSV

---

## Ví Dụ Sử Dụng

### Tìm Best Configuration

1. **Select dataset**: Sidebar → cardio_train
2. **Go to Tab**: "Top Performers"
3. **View top 10**: Sorted by PR-AUC
4. **Result**: Gen1_DecisionTree | none | smote_enn | mutual_info_12

### So Sánh Generations

1. **Sidebar**: Select all generations
2. **Tab**: "Model Comparison"
3. **Observe**: Bar chart shows Gen1 outperforms Gen3/Gen4

### Analyze Preprocessing

1. **Tab**: "Preprocessing Impact"
2. **View**: SMOTE-ENN effect on performance
3. **Finding**:
   - Cardio: SMOTE-ENN ✅ Best
   - Credit: None ✅ Better than SMOTE

---

## Common Actions

| Action | How To |
|--------|--------|
| **Rerun app** | Press `R` key |
| **Refresh data** | Sidebar → Clear cache |
| **Download chart** | Hover → Camera icon 📷 |
| **Export table** | Tab "Detailed Results" → Download CSV |
| **Zoom chart** | Click & drag on chart |
| **Reset zoom** | Double-click on chart |

---

## Troubleshooting

### ❌ "No results found"

```bash
# Run experiments first
python full_comparison.py --data data/raw/cardio_train.csv
```

### ❌ "Module not found: streamlit"

```bash
pip install streamlit plotly
```

### ❌ Port 8501 đã dùng

```bash
# Dùng port khác
streamlit run app_streamlit.py --server.port 8502
```

### 🐌 Dashboard chậm

- Reduce số models selected
- Clear cache: Settings → Clear cache
- Restart Streamlit

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `R` | Rerun app |
| `C` | Clear cache |
| `Ctrl + R` | Reload page |
| `Ctrl + F` | Search in table |

---

## Tips & Tricks

### 💡 Tip 1: Multi-Select
Hold `Ctrl` (Windows) / `Cmd` (Mac) để select nhiều models

### 💡 Tip 2: Compare Datasets
Mở 2 browser tabs:
- Tab 1: cardio_train
- Tab 2: creditcard

### 💡 Tip 3: Export Everything
1. Go to "Detailed Results"
2. Apply filters
3. Download CSV
4. Open in Excel/Python for more analysis

### 💡 Tip 4: Best Performance
Filter by:
- Generation: Chỉ Gen1
- Metric: PR-AUC
→ See simple models win!

---

## Next Steps

1. ✅ Explore all tabs
2. ✅ Try different filters
3. ✅ Compare datasets
4. ✅ Export top configurations
5. ✅ Use findings in your research

**Xem guide đầy đủ tại: [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md)**

---

**Thời gian chạy**: ~5 giây
**URL**: http://localhost:8501
**Tài liệu**: https://docs.streamlit.io
