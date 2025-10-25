# 📊 Hướng Dẫn Đánh Giá Đa Metrics

## 🎯 Tổng Quan

App Streamlit hiện đã hỗ trợ đầy đủ việc đánh giá model theo **nhiều metrics khác nhau**, không chỉ F1 Score!

## 🔧 Các Tính Năng Mới

### 1️⃣ **Chọn Primary Metric (Sidebar)**

Ở sidebar bên trái, bạn có thể chọn metric chính để đánh giá:

- **🎯 F1 Score** - Cân bằng giữa Precision và Recall (mặc định)
- **📊 PR-AUC** - Diện tích dưới đường cong Precision-Recall (tốt cho dữ liệu mất cân bằng)
- **📈 ROC-AUC** - Diện tích dưới đường cong ROC (đánh giá tổng thể)
- **⚖️ Balanced Accuracy** - Trung bình của Sensitivity và Specificity
- **🔍 Sensitivity (Recall)** - Tỉ lệ dương tính thật được phát hiện
- **🛡️ Specificity** - Tỉ lệ âm tính thật được phát hiện
- **🔢 MCC** - Matthews Correlation Coefficient (tốt cho dữ liệu mất cân bằng)

#### Cách Dùng:
1. Mở sidebar
2. Tìm phần **"🎯 Primary Evaluation Metric"**
3. Chọn metric bạn muốn từ dropdown
4. Click vào **"ℹ️ About This Metric"** để xem mô tả chi tiết

### 2️⃣ **Multi-Metric Comparison**

Chọn 2-5 metrics để so sánh trực tiếp:

1. Ở sidebar, tìm **"📊 Multi-Metric Comparison"**
2. Chọn các metrics muốn so sánh (mặc định: PR-AUC, F1, Sensitivity)
3. Dashboard sẽ hiển thị:
   - **Line chart** so sánh các metrics cho top models
   - **Correlation matrix** giữa các metrics
   - **Box plot** phân phối điểm của mỗi metric

### 3️⃣ **Complete Metrics Table**

Bảng so sánh **tất cả metrics** cho top models:

- 🟢 **Green highlight** = Điểm cao nhất cho mỗi metric
- 🟡 **Yellow highlight** = Thời gian huấn luyện nhanh nhất
- Hiển thị: PR-AUC, ROC-AUC, F1, Balanced Accuracy, Sensitivity, Specificity, MCC, NPV, Training Time

### 4️⃣ **Advanced: Custom Weighted Ranking** 🎛️

Tạo công thức đánh giá riêng của bạn:

1. Cuộn xuống phần **"Detailed Analysis & Raw Data"**
2. Mở expander **"🎛️ Advanced: Custom Weighted Ranking"**
3. Sử dụng sliders để gán trọng số cho từng metric:
   - 0.0 = Không tính
   - 1.0 = Trọng số đầy đủ
4. Xem ranking mới dựa trên công thức của bạn
5. Download kết quả với nút **"📥 Download Weighted Rankings CSV"**

#### Ví dụ:
```
Công thức: 0.5×F1 + 0.3×SENSITIVITY + 0.2×SPECIFICITY
```
App sẽ tính điểm tổng hợp và xếp hạng lại các models.

## 📚 Giải Thích Các Metrics

### PR-AUC (Precision-Recall Area Under Curve)
- **Khi nào dùng**: Dataset mất cân bằng (ít positive)
- **Giá trị tốt**: > 0.7
- **Ưu điểm**: Tập trung vào class thiểu số

### ROC-AUC (Receiver Operating Characteristic)
- **Khi nào dùng**: Đánh giá tổng thể, cả hai classes quan trọng
- **Giá trị tốt**: > 0.8
- **Ưu điểm**: Đánh giá khả năng phân biệt

### F1 Score
- **Khi nào dùng**: Cần cân bằng Precision và Recall
- **Giá trị tốt**: > 0.75
- **Ưu điểm**: Tổng hợp precision và recall

### Balanced Accuracy
- **Khi nào dùng**: Dataset mất cân bằng, muốn đối xử công bằng với cả hai classes
- **Giá trị tốt**: > 0.75
- **Ưu điểm**: Trung bình sensitivity và specificity

### Sensitivity (Recall / True Positive Rate)
- **Khi nào dùng**: Quan trọng phải phát hiện hết positives (VD: bệnh ung thư)
- **Giá trị tốt**: > 0.8
- **Ưu điểm**: Giảm False Negatives

### Specificity (True Negative Rate)
- **Khi nào dùng**: Quan trọng không bị false alarm (VD: spam filter)
- **Giá trị tốt**: > 0.8
- **Ưu điểm**: Giảm False Positives

### MCC (Matthews Correlation Coefficient)
- **Khi nào dùng**: Dataset mất cân bằng nghiêm trọng
- **Giá trị**: -1 (tệ) đến +1 (tốt nhất)
- **Ưu điểm**: Metric cân bằng nhất, xem xét tất cả confusion matrix

## 🎯 Khuyến Nghị Sử Dụng

### Cho Dataset Cardio (Tim mạch)
```
Primary: PR-AUC hoặc F1
Compare: [PR-AUC, Sensitivity, Specificity]
Lý do: Quan trọng phát hiện bệnh (Sensitivity cao)
```

### Cho Dataset Credit Card Fraud
```
Primary: PR-AUC hoặc MCC
Compare: [PR-AUC, F1, MCC]
Lý do: Dữ liệu mất cân bằng nghiêm trọng
```

### Cho Nghiên Cứu Tổng Quát
```
Primary: F1
Compare: [F1, ROC-AUC, Balanced Accuracy]
Lý do: Cân bằng tất cả khía cạnh
```

## 📥 Export & Sharing

Bạn có thể download:
1. **Top N models** theo primary metric
2. **All results** - toàn bộ kết quả
3. **Weighted rankings** - kết quả theo công thức custom của bạn

## 💡 Tips & Tricks

1. **So sánh nhanh**: Dùng radar chart (phần Visual Comparison) để thấy ngay model nào cân bằng
2. **Tìm trade-offs**: Xem Multi-Metric line chart để thấy metric nào trade-off với nhau
3. **Hiểu correlation**: Nếu 2 metrics có correlation cao (>0.9), chọn 1 cái thôi
4. **Custom ranking**: Nếu bạn biết rõ domain, tạo weighted formula (VD: y tế → 0.7×Sensitivity + 0.3×Specificity)

## 🚀 Workflow Đề Xuất

1. **Bước 1**: Chọn primary metric phù hợp với use case
2. **Bước 2**: Xem Executive Summary để biết top model
3. **Bước 3**: Check Complete Metrics Table - model có cân bằng không?
4. **Bước 4**: Dùng Multi-Metric Comparison để hiểu sâu hơn
5. **Bước 5**: (Optional) Tạo custom weighted ranking nếu cần
6. **Bước 6**: Download kết quả và document quyết định

## ❓ FAQ

**Q: Tôi nên chọn metric nào?**
A: Phụ thuộc vào:
- Dataset cân bằng? → ROC-AUC hoặc F1
- Dataset mất cân bằng? → PR-AUC hoặc MCC
- Cost của False Negative cao? → Sensitivity
- Cost của False Positive cao? → Specificity

**Q: Tại sao model top 1 của F1 khác với top 1 của PR-AUC?**
A: Vì mỗi metric đo khía cạnh khác nhau. F1 cân bằng precision/recall, PR-AUC nhìn vào toàn bộ precision-recall curve.

**Q: Custom weighted ranking có nên dùng không?**
A: Nên dùng khi bạn:
- Hiểu rõ domain và biết metric nào quan trọng hơn
- Cần trade-off giữa các metrics
- Có yêu cầu cụ thể (VD: Sensitivity ≥ 0.9 AND Specificity ≥ 0.8)

## 📊 Kết Luận

App hiện đã hỗ trợ **đầy đủ** đánh giá đa metrics với:
- ✅ 7+ metrics để chọn
- ✅ Multi-metric comparison tools
- ✅ Custom weighted ranking
- ✅ Visual correlation & distribution
- ✅ Export cho từng cách đánh giá

**Không còn bị giới hạn chỉ F1 nữa!** 🎉
