# 🔬 Hướng Dẫn Chạy Streamlit Dashboard

## Giới Thiệu

Streamlit Dashboard cho phép bạn:
- ✅ Xem và so sánh kết quả 270 experiments
- ✅ Filter theo Generation, Model, Preprocessing
- ✅ Visualize metrics với biểu đồ interactive
- ✅ Phân tích top models và configurations
- ✅ So sánh giữa cardio_train và creditcard datasets

---

## Yêu Cầu

### 1. Kiểm Tra Thư Viện

```bash
pip list | grep streamlit
pip list | grep plotly
```

Nếu chưa có, cài đặt:

```bash
pip install streamlit plotly
```

### 2. Kiểm Tra Dữ Liệu

Dashboard cần file results CSV từ `full_comparison.py`:

```bash
ls experiments/full_comparison/cardio_train/full_comparison_*.csv
ls experiments/full_comparison/creditcard/full_comparison_*.csv
```

Nếu chưa có, chạy experiments trước:

```bash
# Cardio dataset
python full_comparison.py --data data/raw/cardio_train.csv

# Credit card dataset
python full_comparison.py --data data/raw/creditcard.csv
```

---

## Cách Chạy

### Method 1: Chạy Trực Tiếp (Đơn Giản Nhất)

```bash
streamlit run app_streamlit.py
```

Dashboard sẽ tự động mở ở: **http://localhost:8501**

### Method 2: Chỉ Định Port

```bash
streamlit run app_streamlit.py --server.port 8080
```

### Method 3: Chạy Ở Background

```bash
# Windows
start streamlit run app_streamlit.py

# Linux/Mac
nohup streamlit run app_streamlit.py &
```

### Method 4: Network Access (Truy cập từ máy khác)

```bash
streamlit run app_streamlit.py --server.address 0.0.0.0
```

Sau đó truy cập: `http://<your-ip>:8501`

---

## Cấu Trúc Dashboard

### 📁 **Sidebar - Settings**

#### 1. Dataset Selection
- **cardio_train**: Cardiovascular disease dataset (balanced)
- **creditcard**: Credit card fraud detection (highly imbalanced)

#### 2. Model Selection
- Filter by **Generation** (1, 2, 3, 4)
- Select specific **Models**
- Default: Hiển thị tất cả

#### 3. Preprocessing Filters
- **Scaler**: standard, minmax, robust, none
- **Imbalance**: none, smote, adasyn, smote_enn
- **Feature Selection**: none, select_k_best_5/12, mutual_info_5/12

#### 4. Metric Selection
- **Primary metric**: PR-AUC (default)
- Other options: Accuracy, F1, ROC-AUC, Sensitivity, Specificity, MCC

---

### 📊 **Main Content - Tabs**

#### **Tab 1: Overview**
- 📈 **Key Metrics Cards**
  - Total experiments
  - Best PR-AUC
  - Best model
  - Average training time

- 📊 **Overall Statistics**
  - Summary table với mean/std/min/max

#### **Tab 2: Model Comparison**
- 📊 **Interactive Bar Charts**
  - Metrics comparison across selected models
  - Group by generation

- 📈 **Performance Distribution**
  - Box plots cho mỗi metric
  - Scatter plots: PR-AUC vs F1, etc.

#### **Tab 3: Top Performers**
- 🏆 **Top 10 Configurations**
  - Sorted by selected metric
  - Show full config details

- 🔝 **Best per Generation**
  - Gen1, Gen2, Gen3, Gen4 winners

#### **Tab 4: Preprocessing Impact**
- 📊 **Scaler Impact**
  - Average performance by scaler type

- 📊 **Imbalance Handling**
  - SMOTE vs ADASYN vs SMOTE-ENN vs None

- 📊 **Feature Selection**
  - Impact of k=5 vs k=12
  - Mutual info vs SelectKBest

#### **Tab 5: Detailed Results**
- 📋 **Full Results Table**
  - Searchable, sortable, filterable
  - Export to CSV

- 📊 **Correlation Heatmap**
  - Metrics correlation analysis

---

## Tính Năng Chính

### 1. **Interactive Filtering**
```
Sidebar → Select filters → Charts update real-time
```

### 2. **Hover for Details**
Hover lên biểu đồ để xem:
- Exact values
- Model configuration
- Standard deviation

### 3. **Download Charts**
Mỗi biểu đồ có nút 📷 để download PNG

### 4. **Export Data**
Download filtered results as CSV

### 5. **Responsive Layout**
- Wide layout for better visualization
- Adaptive to screen size

---

## Ví Dụ Sử Dụng

### Scenario 1: Tìm Best Model cho Cardio Dataset

1. **Start dashboard**:
   ```bash
   streamlit run app_streamlit.py
   ```

2. **Sidebar → Dataset**: Select `cardio_train`

3. **Tab "Top Performers"**: Xem top 10 configurations

4. **Result**:
   ```
   Best: Gen1_DecisionTree
   Config: none | smote_enn | mutual_info_12
   PR-AUC: 0.8023
   ```

### Scenario 2: So Sánh Gen3 vs Gen4

1. **Sidebar → Generation**: Chọn [3, 4]

2. **Tab "Model Comparison"**: Xem bar chart

3. **Analysis**:
   - Gen3 (XGBoost, LightGBM, CatBoost) performance
   - Gen4 (PyTorch MLP, TabNet) performance
   - Training time comparison

### Scenario 3: Phân Tích SMOTE Impact

1. **Tab "Preprocessing Impact"**

2. **Section "Imbalance Handling"**

3. **Observe**:
   - Cardio: SMOTE-ENN tốt nhất
   - Credit Card: None tốt hơn SMOTE

### Scenario 4: Export Top 20 Models

1. **Tab "Detailed Results"**

2. **Sort by PR-AUC** (click column header)

3. **Download** filtered CSV

---

## Troubleshooting

### Lỗi: "No results found"

**Nguyên nhân**: Chưa có file CSV results

**Giải pháp**:
```bash
python full_comparison.py --data data/raw/cardio_train.csv
```

### Lỗi: "Module not found: streamlit"

**Giải pháp**:
```bash
pip install streamlit plotly
```

### Lỗi: "Port 8501 already in use"

**Giải pháp**:
```bash
# Dùng port khác
streamlit run app_streamlit.py --server.port 8502

# Hoặc kill process
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8501 | xargs kill -9
```

### Dashboard Chạy Chậm

**Giải pháp**:
- Reduce số models được select
- Clear cache: Settings → Clear cache
- Restart Streamlit

---

## Shortcuts & Tips

### Keyboard Shortcuts (trong browser)

| Key | Action |
|-----|--------|
| `R` | Rerun app |
| `Ctrl + R` | Reload page |
| `Ctrl + Shift + I` | Open DevTools |

### Performance Tips

1. **Use filters wisely**
   - Don't select all models at once for large datasets
   - Filter by generation first

2. **Cache is your friend**
   - Data is cached automatically
   - Only reloads when file changes

3. **Export important views**
   - Download charts as PNG
   - Export filtered tables as CSV

### Customization

Edit `app_streamlit.py` để customize:

```python
# Change default dataset
dataset_choice = st.sidebar.radio(
    "Select Dataset",
    options=['cardio_train', 'creditcard'],
    index=0  # ← Change to 1 for creditcard default
)

# Change default metric
metric_choice = st.sidebar.selectbox(
    "Primary Metric",
    options=['pr_auc', 'accuracy', 'f1', ...],
    index=0  # ← Change index
)
```

---

## Advanced Usage

### 1. Run on Remote Server

```bash
# On server
streamlit run app_streamlit.py --server.address 0.0.0.0 --server.port 8501

# Access from local machine
http://<server-ip>:8501
```

### 2. Password Protection

Create `.streamlit/config.toml`:
```toml
[server]
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

### 3. Auto-reload on Data Change

```bash
# Enable auto-reload
streamlit run app_streamlit.py --server.fileWatcherType poll
```

### 4. Deploy to Cloud

**Streamlit Cloud** (Free):
1. Push code to GitHub
2. Visit: https://share.streamlit.io
3. Connect repository
4. Deploy!

**Heroku**:
```bash
# Create Procfile
echo "web: streamlit run app_streamlit.py --server.port $PORT" > Procfile

# Deploy
heroku create
git push heroku main
```

---

## File Structure

```
cls_review/
├── app_streamlit.py              # ← Main dashboard app
│
├── experiments/
│   └── full_comparison/
│       ├── cardio_train/
│       │   └── full_comparison_*.csv    # ← Data source 1
│       └── creditcard/
│           └── full_comparison_*.csv    # ← Data source 2
│
└── .streamlit/
    └── config.toml               # ← Optional config
```

---

## Screenshots & Demo

### Main Dashboard
```
╔══════════════════════════════════════════════════════════╗
║  🔬 ML Models Comparison Dashboard                      ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  ✅ Loaded: full_comparison_20251018_022851.csv         ║
║                                                          ║
║  ┌─────────┬──────────┬──────────┬──────────┐          ║
║  │ Total   │ Best     │ Best     │ Avg      │          ║
║  │ Exps    │ PR-AUC   │ Model    │ Time     │          ║
║  ├─────────┼──────────┼──────────┼──────────┤          ║
║  │ 270     │ 0.8023   │ DTree    │ 238.3s   │          ║
║  └─────────┴──────────┴──────────┴──────────┘          ║
║                                                          ║
║  [Tab: Overview | Model Comparison | Top Performers]   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

---

## Quick Reference

### Common Commands

```bash
# Start dashboard
streamlit run app_streamlit.py

# Different port
streamlit run app_streamlit.py --server.port 8080

# Remote access
streamlit run app_streamlit.py --server.address 0.0.0.0

# Check version
streamlit --version

# Clear cache
# (In dashboard: Settings → Clear cache)
```

### URLs

- **Local**: http://localhost:8501
- **Network**: http://<your-ip>:8501
- **Docs**: https://docs.streamlit.io

---

## FAQ

**Q: Dashboard không update khi tôi chạy experiment mới?**

A: Click nút "R" (rerun) hoặc refresh browser.

**Q: Làm sao để so sánh 2 datasets?**

A: Mở 2 tabs browser:
- Tab 1: Select cardio_train
- Tab 2: Select creditcard

**Q: Export tất cả biểu đồ cùng lúc?**

A: Hiện tại phải download từng biểu đồ. Hoặc dùng browser Print → Save as PDF.

**Q: Dashboard có hỗ trợ mobile không?**

A: Có, responsive design. Nhưng desktop experience tốt hơn.

**Q: Làm sao customize theme?**

A: Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

---

## Next Steps

1. ✅ Chạy dashboard
2. ✅ Explore các tabs
3. ✅ Filter và analyze results
4. ✅ Export findings
5. ✅ Tích hợp vào presentation

**Happy Analyzing! 🚀**

---

**Last Updated**: 2025-10-19
**Version**: 1.0
**Support**: Check app_streamlit.py for code details
