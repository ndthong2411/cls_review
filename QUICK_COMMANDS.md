# 🚀 Quick Commands Reference

## 📋 Bảng Lệnh Tổng Hợp

### 🏃 Training & Running

| Mục đích | Lệnh |
|----------|------|
| **Chạy training đầy đủ** | `python full_comparison.py` |
| **Chạy training (bỏ cache)** | `python full_comparison.py --no-cache` |
| **Xem các experiments đã cache** | `python full_comparison.py --list-cache` |
| **Xóa toàn bộ cache** | `python full_comparison.py --clear-cache` |
| **Xem hướng dẫn** | `python full_comparison.py --help` |
| **Phân tích kết quả** | `python analyze_results.py` |
| **🎨 Chạy Streamlit Dashboard** | `streamlit run app_streamlit.py` |

---

### 📝 Xem & Quản Lý Logs

| Mục đích | Lệnh |
|----------|------|
| **Xem log mới nhất (50 dòng)** | `Get-Content experiments/logs/*.log -Tail 50` |
| **Follow log real-time** | `Get-Content experiments/logs/training_*.log -Wait -Tail 30` |
| **List tất cả logs** | `Get-ChildItem experiments/logs/*.log` |
| **Tìm errors trong logs** | `Select-String -Path "experiments/logs/*.log" -Pattern "Error\|Failed\|Exception"` |
| **Xem log trong 24h** | `Get-ChildItem experiments/logs/*.log \| Where-Object {$_.LastWriteTime -gt (Get-Date).AddHours(-24)}` |
| **Xóa logs cũ >7 ngày** | `Get-ChildItem experiments/logs/*.log \| Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} \| Remove-Item` |

---

### 📊 Xem Results

| Mục đích | Lệnh |
|----------|------|
| **Xem CSV results** | `Import-Csv experiments/full_comparison/full_comparison_*.csv \| Select-Object -First 10` |
| **Top 10 models** | `Import-Csv experiments/full_comparison/full_comparison_*.csv \| Sort-Object pr_auc -Descending \| Select-Object -First 10` |
| **Tìm Gen3 results** | `Import-Csv experiments/full_comparison/full_comparison_*.csv \| Where-Object {$_.model -like "Gen3*"}` |
| **So sánh generations** | `Import-Csv experiments/full_comparison/full_comparison_*.csv \| Group-Object generation` |

---

### 💾 Quản Lý Cache

| Mục đích | Lệnh |
|----------|------|
| **Xem cache size** | `(Get-ChildItem experiments/model_cache/*.pkl \| Measure-Object -Property Length -Sum).Sum / 1MB` |
| **Đếm số models cached** | `(Get-ChildItem experiments/model_cache/*.pkl).Count` |
| **Xóa cache Gen1** | `Remove-Item experiments/model_cache/Gen1_*.pkl` |
| **Xóa cache cũ >30 ngày** | `Get-ChildItem experiments/model_cache/*.pkl \| Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-30)} \| Remove-Item` |

---

### 🗂️ Quản Lý Files

| Mục đích | Lệnh |
|----------|------|
| **Xem cấu trúc thư mục** | `tree /F experiments` |
| **List models đã train** | `Get-ChildItem experiments/full_comparison/best_model/` |
| **Backup results** | `Copy-Item experiments/full_comparison/*.csv -Destination backup/` |

---

## 🔥 One-Liners Hữu Ích

### Quick Analysis
```powershell
# Summary by generation
Import-Csv experiments/full_comparison/full_comparison_*.csv | 
  Group-Object generation | 
  ForEach-Object {
    [PSCustomObject]@{
      Generation = $_.Name
      Count = $_.Count
      AvgPR_AUC = ($_.Group.pr_auc | Measure-Object -Average).Average
      MaxPR_AUC = ($_.Group.pr_auc | Measure-Object -Maximum).Maximum
    }
  } | Format-Table -AutoSize
```

### Monitor Training
```powershell
# Watch training progress
while($true) {
  Clear-Host
  Get-Content experiments/logs/training_*.log -Tail 20
  Start-Sleep -Seconds 5
}
```

### Find Best Config
```powershell
# Best model with config details
Import-Csv experiments/full_comparison/full_comparison_*.csv | 
  Sort-Object pr_auc -Descending | 
  Select-Object -First 1 | 
  Format-List
```

---

## 📌 Aliases Hữu Ích

Thêm vào PowerShell profile (`$PROFILE`):

```powershell
function train { python full_comparison.py }
function analyze { python analyze_results.py }
function taillog { Get-Content experiments/logs/*.log -Tail 50 }
function followlog { Get-Content experiments/logs/training_*.log -Wait }
function results { Import-Csv experiments/full_comparison/full_comparison_*.csv | Sort-Object pr_auc -Descending | Select-Object -First 10 }
```

---

## 🎯 Common Workflows

### 1. Chạy Training Mới
```powershell
# Clear old cache và chạy lại
python full_comparison.py --clear-cache
python full_comparison.py
```

### 2. Re-train Với Cache
```powershell
# Sử dụng cache để nhanh hơn
python full_comparison.py
```

### 3. Phân Tích Results
```powershell
# Xem kết quả và phân tích
python analyze_results.py
Import-Csv experiments/full_comparison/full_comparison_*.csv | Select-Object -First 20
```

### 4. Monitor Long Run
```powershell
# Terminal 1: Run training
python full_comparison.py

# Terminal 2: Follow log
Get-Content experiments/logs/training_*.log -Wait -Tail 30
```

---

## 📁 Cấu Trúc Thư Mục

```
cls_review/
├── README.md                      # Main documentation
├── QUICK_COMMANDS.md             # This file
├── full_comparison.py            # Main training script
├── analyze_results.py            # Results analysis
├── requirements.txt              # Dependencies
├── docs/                         # Documentation
│   ├── INDEX.md
│   ├── LOGGING_GUIDE.md
│   ├── CHANGELOG_OCT16.md
│   └── ...
└── experiments/
    ├── logs/                     # Training logs
    ├── model_cache/              # Cached experiments
    ├── full_comparison/          # Results & best model
    └── results_summary.csv
```

---

## 💡 Tips & Tricks

### 1. Backup Before Major Changes
```powershell
# Backup cache và results
$date = Get-Date -Format yyyyMMdd
Copy-Item experiments/model_cache -Destination "backup/cache_$date" -Recurse
Copy-Item experiments/full_comparison/*.csv -Destination "backup/"
```

### 2. Clean Old Files
```powershell
# Archive logs >7 days
New-Item -Path "experiments/logs/archive" -ItemType Directory -Force
Get-ChildItem experiments/logs/*.log | 
  Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} |
  Move-Item -Destination "experiments/logs/archive/"
```

### 3. Compare Two Runs
```powershell
# Extract top models from 2 runs
$run1 = Import-Csv experiments/full_comparison/full_comparison_20251016_094856.csv
$run2 = Import-Csv experiments/full_comparison/full_comparison_20251016_142547.csv

Compare-Object ($run1 | Select-Object -First 10) ($run2 | Select-Object -First 10) -Property model, pr_auc
```

---

**Cập nhật**: October 16, 2025  
**Version**: 2.0  
**Documentation**: See `docs/INDEX.md` for full documentation
