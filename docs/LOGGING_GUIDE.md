# 📝 Training Logs Guide

## 📍 Vị Trí Log Files

Tất cả training logs được tự động lưu tại:
```
experiments/logs/training_YYYYMMDD_HHMMSS.log
```

### Ví Dụ:
```
experiments/
  └── logs/
      ├── training_20251016_142547.log
      ├── training_20251016_151230.log
      └── training_20251016_163045.log
```

---

## 📋 Nội Dung Log File

Mỗi log file chứa đầy đủ thông tin:

### 1. **Thông Tin Chung**
- Start time
- Dataset info (số samples, features, class balance)
- Train/test split details
- Cache status

### 2. **Model Configuration**
- 13 models loaded (Gen1-4)
- GPU status (XGBoost, LightGBM, CatBoost)
- Total experiments to run

### 3. **Training Progress**
Cho mỗi experiment (108 total):
```
[79/108] Gen3_LightGBM | Scale: none | Imb: none | FeatSel: none
  ✓ PR-AUC: 0.7855 | Sens: 0.6982 | Spec: 0.7743 | F1: 0.7257 | Time: 22.7s
```

### 4. **Final Summary**
- Top 10 models ranked by PR-AUC
- Generation comparison (best config per generation)
- Total training time
- Best model details

---

## 🔍 Xem Log Files

### Xem Log Mới Nhất:
```powershell
Get-Content experiments/logs/training_*.log -Tail 50
```

### Xem File Log Cụ Thể:
```powershell
Get-Content experiments/logs/training_20251016_142547.log
```

### Tìm Log Trong 24 Giờ Qua:
```powershell
Get-ChildItem experiments/logs/*.log | 
  Where-Object {$_.LastWriteTime -gt (Get-Date).AddHours(-24)} | 
  Select-Object Name, LastWriteTime
```

### Grep Trong Logs:
```powershell
# Tìm Gen3 results
Select-String -Path "experiments/logs/*.log" -Pattern "Gen3_" 

# Tìm errors
Select-String -Path "experiments/logs/*.log" -Pattern "Error|Failed|Exception"
```

---

## 📊 Log Analysis

### Extract Top Models:
```powershell
Select-String -Path "experiments/logs/training_*.log" -Pattern "^\[.*PR-AUC: 0\.[89]" | 
  Sort-Object | Select-Object -Last 20
```

### Count Experiments:
```powershell
(Select-String -Path "experiments/logs/training_20251016_142547.log" -Pattern "^\[\d+/\d+\]").Count
```

### Check Training Status:
```powershell
# Xem dòng cuối cùng để biết status
Get-Content experiments/logs/training_*.log -Tail 1
```

---

## 🛠️ Tính Năng Log System

### ✅ Automatic Features:
1. **Auto-created** - Log file tự động tạo khi chạy `full_comparison.py`
2. **Timestamped** - Tên file có timestamp để tránh ghi đè
3. **Real-time** - Ghi ngay khi training (không chờ kết thúc)
4. **Dual output** - Hiện cả trên console VÀ ghi file
5. **Flush immediately** - Không bị mất log nếu crash

### 📝 Log Format:
- **UTF-8 encoding** - Hỗ trợ tiếng Việt
- **Structured sections** - Dễ parse và tìm kiếm
- **Metrics included** - PR-AUC, Sens, Spec, F1, Time
- **Progress tracking** - [N/108] cho mỗi experiment

---

## 🚨 Troubleshooting

### Log File Không Tồn Tại?
```powershell
# Kiểm tra thư mục logs
Test-Path experiments/logs
```

Nếu không tồn tại:
```powershell
New-Item -Path "experiments/logs" -ItemType Directory -Force
```

### Log Bị Cắt Giữa Chừng?
- **Nguyên nhân**: Training bị interrupt (Ctrl+C, crash, out of memory)
- **Giải pháp**: Xem file log để biết dừng ở experiment nào, resume từ đó

### Không Thấy Log Mới?
```powershell
# Xem file log mới nhất
Get-ChildItem experiments/logs/*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

---

## 📌 Best Practices

### 1. **Archive Old Logs**
```powershell
# Move logs cũ hơn 7 ngày
Get-ChildItem experiments/logs/*.log | 
  Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} |
  Move-Item -Destination "experiments/logs/archive/"
```

### 2. **Compare Logs**
```powershell
# So sánh 2 lần chạy
$log1 = "experiments/logs/training_20251016_094856.log"
$log2 = "experiments/logs/training_20251016_142547.log"

# Extract top models
Select-String -Path $log1 -Pattern "Best PR-AUC"
Select-String -Path $log2 -Pattern "Best PR-AUC"
```

### 3. **Monitor Long Runs**
```powershell
# Tail -f equivalent (follow log)
Get-Content experiments/logs/training_20251016_142547.log -Wait
```

---

## 🎯 Quick Reference

| Action | Command |
|--------|---------|
| Xem log mới nhất | `Get-Content experiments/logs/*.log -Tail 50` |
| List tất cả logs | `Get-ChildItem experiments/logs/*.log` |
| Tìm errors | `Select-String -Path "experiments/logs/*.log" -Pattern "Error"` |
| Follow log | `Get-Content <log_file> -Wait` |
| Xóa old logs | `Remove-Item experiments/logs/*.log -Exclude "*$(Get-Date -Format yyyyMMdd)*"` |

---

**Cập nhật**: October 16, 2025  
**Log Directory**: `experiments/logs/`  
**Naming**: `training_YYYYMMDD_HHMMSS.log`
