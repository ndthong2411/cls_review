# 🎨 Streamlit Dashboard - User Guide

## 🚀 Quick Start

### Chạy Dashboard:
```powershell
streamlit run app_streamlit.py
```

Dashboard sẽ mở tự động tại: **http://localhost:8501**

---

## 📊 Tính Năng Dashboard

### 1️⃣ **Overview Tab** - Tổng Quan
- ✅ Thống kê tổng thể (số experiments, models, generations)
- ✅ Metrics tốt nhất đạt được
- ✅ Performance by generation
- ✅ Training time statistics
- ✅ Best model information chi tiết

### 2️⃣ **Top Models Tab** - Top Models
- ✅ Biểu đồ so sánh metrics (PR-AUC, Sensitivity, Specificity, F1, ROC-AUC, MCC)
- ✅ Radar chart đa chiều (so sánh tối đa 5 models)
- ✅ Bảng top N models với gradient coloring
- ✅ Filter theo generation

### 3️⃣ **Generation Analysis Tab** - Phân Tích Generation
- ✅ Performance comparison giữa các generations
- ✅ Box plots phân phối PR-AUC và training time
- ✅ Thống kê chi tiết (mean, std, min, max)
- ✅ Error bars cho độ tin cậy

### 4️⃣ **Preprocessing Impact Tab** - Ảnh Hưởng Preprocessing
- ✅ Top 15 preprocessing configurations tốt nhất
- ✅ Component-wise analysis:
  - Scaler impact (Standard, Robust, MinMax, None)
  - Imbalance handling impact (SMOTE, SMOTE-ENN, None)
  - Feature selection impact (SelectKBest, None)
- ✅ Interactive bar charts

### 5️⃣ **Performance Analysis Tab** - Phân Tích Hiệu Suất
- ✅ Performance vs Training Time scatter plot
- ✅ Efficiency score (Performance / Time)
- ✅ Metrics correlation heatmap
- ✅ Scatter matrix (pairwise metrics correlation)

### 6️⃣ **Detailed Data Tab** - Dữ Liệu Chi Tiết
- ✅ Full data table với filters:
  - Filter by scaler
  - Filter by imbalance method
  - Filter by feature selection
  - Filter by generation
- ✅ Sort by any metric (ascending/descending)
- ✅ Gradient coloring cho metrics
- ✅ **Download CSV** của filtered data

---

## 🎛️ Sidebar Controls

### Model Selection
- **Filter by Generation**: Chọn Gen1, Gen2, Gen3, Gen4
- **Select Models**: Chọn models cụ thể để so sánh (multiselect)

### Settings
- **Primary Metric**: Chọn metric chính (PR-AUC, Sensitivity, Specificity, F1, ROC-AUC, MCC)
- **Top N Models**: Slider để chọn số lượng top models hiển thị (5-50)

---

## 📈 Visualizations

### Interactive Charts (Plotly)
1. **Bar Charts** - So sánh metrics giữa models/generations
2. **Radar Charts** - Đa chiều performance comparison
3. **Box Plots** - Phân phối metrics
4. **Scatter Plots** - Performance vs Time, correlation analysis
5. **Heatmaps** - Metrics correlation
6. **Scatter Matrix** - Pairwise relationships

### Features
- ✅ **Interactive**: Hover để xem chi tiết
- ✅ **Zoom/Pan**: Phóng to/thu nhỏ charts
- ✅ **Download**: Export charts as PNG
- ✅ **Legend toggle**: Ẩn/hiện data series

---

## 💡 Use Cases

### 1. So Sánh Models
```
1. Sidebar → Select Models → Chọn models muốn so sánh
2. Tab "Top Models" → Xem metrics comparison
3. Scroll down → Xem radar chart
```

### 2. Phân Tích Generation
```
1. Tab "Generation Analysis"
2. Xem performance charts
3. Check box plots để thấy distribution
```

### 3. Tìm Best Preprocessing Config
```
1. Tab "Preprocessing Impact"
2. Xem top 15 configs
3. Analyze component-wise impact
```

### 4. Tìm Model Hiệu Quả Nhất
```
1. Tab "Performance Analysis"
2. Xem efficiency score chart
3. Tìm balance giữa performance và speed
```

### 5. Export Data
```
1. Tab "Detailed Data"
2. Apply filters
3. Sort by metric
4. Click "Download Filtered Data as CSV"
```

---

## 🎨 UI Features

### Color Coding
- **Gradient Coloring**: Metrics tables có màu gradient (đỏ → vàng → xanh)
- **Color Scales**: 
  - Viridis (preprocessing)
  - RdYlGn (metrics)
  - Set2 (generations)

### Responsive Layout
- ✅ Wide mode layout
- ✅ Columns auto-adjust
- ✅ Mobile-friendly

### Interactive Elements
- ✅ Multiselect dropdowns
- ✅ Sliders
- ✅ Radio buttons
- ✅ Download buttons

---

## 🔧 Advanced Usage

### Custom Filters
```python
# Trong tab "Detailed Data"
1. Filter by scaler: Standard, Robust, MinMax
2. Filter by imbalance: SMOTE, SMOTE-ENN
3. Filter by feature selection: SelectKBest
4. Sort by PR-AUC descending
5. Download filtered results
```

### Compare Specific Configs
```python
# Sidebar
1. Select Generation: Gen3 only
2. Select Models: Gen3_XGBoost, Gen3_LightGBM, Gen3_CatBoost
3. Tab "Top Models" → Compare metrics
```

### Find Trade-offs
```python
# Tab "Performance Analysis"
1. Scatter plot: Xác định models với high PR-AUC và low training time
2. Efficiency score: Tìm most efficient model
```

---

## 📊 Metrics Explained

| Metric | Ý Nghĩa | Range |
|--------|---------|-------|
| **PR-AUC** | Precision-Recall Area Under Curve (primary) | 0-1 |
| **Sensitivity** | Recall, True Positive Rate | 0-1 |
| **Specificity** | True Negative Rate | 0-1 |
| **F1-Score** | Harmonic mean of Precision & Recall | 0-1 |
| **ROC-AUC** | Receiver Operating Characteristic AUC | 0-1 |
| **MCC** | Matthews Correlation Coefficient | -1 to 1 |
| **NPV** | Negative Predictive Value | 0-1 |

---

## 🚨 Troubleshooting

### Dashboard không load
```powershell
# Check if results exist
Test-Path experiments/full_comparison/full_comparison_*.csv

# If not, run training first
python full_comparison.py
```

### Lỗi "No results found"
```powershell
# Chạy training để tạo results
python full_comparison.py
```

### Charts không hiển thị
```powershell
# Install plotly
pip install plotly kaleido
```

### Streamlit lỗi
```powershell
# Reinstall streamlit
pip install --upgrade streamlit
```

---

## 🎯 Best Practices

1. **Chọn Metrics Phù Hợp**: 
   - Medical: PR-AUC, Sensitivity
   - Balanced: F1-Score, ROC-AUC

2. **So Sánh Ít Models**: 
   - Radar chart: ≤5 models (rõ ràng)
   - Bar chart: ≤10 models (dễ đọc)

3. **Filter Thông Minh**:
   - Filter by generation trước
   - Sau đó chọn specific models

4. **Export Data**:
   - Save filtered results cho analysis sâu hơn
   - Compare với previous runs

---

## 📱 Shortcuts

| Action | Shortcut |
|--------|----------|
| Refresh dashboard | `Ctrl + R` |
| Clear cache | `C` |
| Fullscreen chart | Click chart → Camera icon |
| Download chart | Hover chart → Camera icon |

---

## 🎓 Tips

💡 **Tip 1**: Dùng "Top N Models" slider để focus vào top performers  
💡 **Tip 2**: Filter by Generation 3 để xem advanced models only  
💡 **Tip 3**: Check efficiency score để tìm fast models với good performance  
💡 **Tip 4**: Download filtered data để làm report  
💡 **Tip 5**: Radar chart tốt nhất để so sánh tổng thể performance  

---

## 📚 Related Docs

- `docs/LOGGING_GUIDE.md` - Training logs
- `QUICK_COMMANDS.md` - Command reference
- `README.md` - Main documentation

---

**Version**: 1.0  
**Last Updated**: October 16, 2025  
**Port**: http://localhost:8501
