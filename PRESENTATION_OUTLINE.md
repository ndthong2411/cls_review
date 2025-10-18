# BÀI THUYẾT TRÌNH: DỰ ĐOÁN SUY GIẢM NHẬN THỨC Ở BỆNH NHÂN PARKINSON
## Tích hợp: Paper Analysis + Thực nghiệm Cardiovascular Disease

---

## **PHẦN 1: GIỚI THIỆU** (2-3 phút)

### Slide 1: Title
- Tiêu đề: "Predicting Cognitive Decline in Parkinson's Disease Using Machine Learning"
- Tên người thuyết trình
- Vai trò: IT Researcher - Focus on ML Models

### Slide 2: Context
- Parkinson's Disease (PD) và vấn đề Mild Cognitive Impairment (MCI)
- Tầm quan trọng của dự đoán sớm
- **Mục tiêu**: So sánh nhiều ML models để tìm phương pháp tốt nhất

---

## **PHẦN 2: PAPER PARKINSON - METHODS** (5-7 phút)

### Slide 3: Dataset & Features
- **PPMI Study**: 423 patients, 4 years follow-up
- **Features** (40 predictors):
  - Clinical: Age, motor symptoms (UPDRS), cognitive tests (MoCA)
  - Biomarkers: CSF (Aβ42, tau), neuroimaging (hippocampus volume)
- **Target**: PD-MCI vs PD-normal

### Slide 4: Models Compared in Paper
**6 Models tested:**
| Model | Type | Complexity |
|-------|------|-----------|
| Logistic Regression | Linear | Low |
| Support Vector Machine | Kernel-based | Medium |
| Random Forest | Ensemble | Medium |
| Gradient Boosting | Ensemble | Medium-High |
| Neural Network | Deep Learning | High |
| Ensemble Methods | Combined | High |

### Slide 5: Evaluation Metrics
- **AUC-ROC**: Primary metric
- **Sensitivity/Specificity**: Clinical utility
- **Calibration**: Prediction reliability
- **Feature Importance**: Interpretability

---

## **PHẦN 3: THỰC NGHIỆM MINH HỌA** (8-10 phút) ⭐ **PHẦN CHÍNH**

### Slide 6: Experimental Design - Progressive Model Evolution
> "Để hiểu sâu hơn về model selection, tôi đã thực hiện thực nghiệm tương tự trên **Cardiovascular Disease prediction**"

**Dataset**: 70,000 patients, similar medical prediction task

**Progressive Framework (4 Generations):**
- **Gen 1 (Baseline)**: Logistic Regression, Decision Tree, KNN
- **Gen 2 (Ensemble)**: Random Forest, Extra Trees, GradientBoosting, SVM, MLP
- **Gen 3 (Advanced Boosting)**: XGBoost, LightGBM, CatBoost
- **Gen 4 (Deep Learning)**: PyTorch MLP, TabNet

**Total**: 11 models × ~25 preprocessing configs = **270 experiments**

### Slide 7: Preprocessing Strategies Tested
**3 Preprocessing Dimensions:**

1. **Scaling** (3 methods):
   - Standard Scaler
   - Robust Scaler
   - None

2. **Imbalance Handling** (3 methods):
   - None
   - SMOTE (over-sampling)
   - SMOTE-ENN (hybrid)

3. **Feature Selection** (5 methods):
   - None
   - SelectKBest (k=5, k=12)
   - Mutual Information (k=5, k=12)

### Slide 8: Results - Generation Comparison
**BIỂU ĐỒ 1**: `1_generation_comparison.png`

| Generation | Mean PR-AUC | Best Model | Best AUC |
|------------|-------------|------------|----------|
| Gen 1 | 0.7576 | DecisionTree | 0.8023 |
| Gen 2 | 0.7653 | GradientBoosting | 0.7865 |
| Gen 3 | 0.7711 | CatBoost | 0.7864 |
| Gen 4 | 0.7660 | PyTorch MLP | 0.7839 |

**Key Findings:**
- ✅ Gen 3 (Advanced Boosting) có mean performance cao nhất
- ⚠️ Gen 4 (Deep Learning) **KHÔNG** tốt hơn (dataset size limitation)
- 🚀 Simple models (Gen 1) với **proper preprocessing** có thể vượt mặt complex models!

### Slide 9: Preprocessing Impact
**BIỂU ĐỒ 2**: `2_preprocessing_impact.png`

**Best Configurations:**
- **Scaling**: NONE (PR-AUC: 0.7691)
  → Tree-based models không cần scaling!

- **Imbalance**: SMOTE-ENN (PR-AUC: 0.7678)
  → Hybrid approach tốt nhất cho medical data

- **Feature Selection**: Mutual Info (k=12) (PR-AUC: 0.7723)
  → Trade-off giữa performance và interpretability

### Slide 10: Top 10 Configurations
**BIỂU ĐỒ 3**: `3_top10_models.png`

**Best Overall:**
- Model: **Decision Tree** (Gen 1)
- Config: **No scaling + SMOTE-ENN + Mutual Info (k=12)**
- **PR-AUC**: 0.8023 ± 0.0041
- **Sensitivity**: 0.6900 (important for medical screening)
- **Specificity**: 0.7662
- **Training Time**: 22s (very fast!)

**Top 3:**
1. DecisionTree + SMOTE-ENN + MutualInfo-12: **0.8023**
2. KNN + SMOTE-ENN + SelectKBest-12: **0.8022**
3. KNN + SMOTE-ENN + MutualInfo-12: **0.8011**

**Insight**: All top 10 sử dụng **SMOTE-ENN** for imbalance handling!

### Slide 11: Performance vs Training Time
**BIỂU ĐỒ 4**: `4_performance_vs_time.png`

**Trade-offs:**
- **Fast + Good**: Decision Tree (22s, AUC=0.8023)
- **Slow + Good**: CatBoost (513s, AUC=0.7864)
- **Very Slow + Good**: PyTorch MLP (310s, AUC=0.7839)

**Recommendation**: Choose based on deployment constraints
- Clinical practice: Fast models (Gen 1-2)
- Research: Advanced models (Gen 3-4) for best accuracy

---

## **PHẦN 4: PAPER RESULTS** (3-5 phút)

### Slide 12: Paper Parkinson - Results
**Best Models from Paper:**

| Model | AUC | Sensitivity | Specificity |
|-------|-----|-------------|-------------|
| Random Forest | 0.78-0.82 | 0.75-0.80 | 0.70-0.75 |
| Gradient Boosting | 0.76-0.80 | 0.72-0.78 | 0.68-0.73 |
| Ensemble | 0.80-0.83 | 0.78-0.82 | 0.73-0.78 |

**Most Important Features:**
1. Baseline cognitive scores (MoCA, fluency)
2. CSF biomarkers (Aβ42, tau)
3. Motor severity (UPDRS)
4. Age, education

### Slide 13: Comparison - Paper vs My Experiment
| Aspect | Paper (Parkinson) | My Experiment (CVD) |
|--------|-------------------|---------------------|
| Dataset Size | 423 patients | 70,000 patients |
| Features | 40 (clinical + biomarkers) | 15 (clinical) |
| Models | 6 models | 11 models (4 generations) |
| Best AUC | 0.80-0.83 | 0.7790 (ROC), 0.8023 (PR) |
| Best Model | Ensemble | Decision Tree + SMOTE-ENN |
| Key Finding | Biomarkers boost accuracy | Preprocessing > Model complexity |

**SO SÁNH:**
- ✅ My experiment AUC: 0.7790-0.8023 → **TƯƠNG ĐƯƠNG** với paper!
- 💡 Paper chưa test advanced boosting (XGBoost, LightGBM, CatBoost)
- 🔬 Paper chưa systematic preprocessing comparison

---

## **PHẦN 5: ĐỊNH HƯỚNG - CONTINUAL LEARNING** (5-7 phút) ⭐ **TRỌNG TÂM**

### Slide 14: Current Limitations
**Vấn đề với Current Approaches:**
1. **Single-site data**: Model chỉ học từ 1 bệnh viện
2. **Static models**: Không update khi có data mới
3. **Data privacy**: Không share patient data giữa các sites
4. **Distribution shift**: Data khác nhau giữa các trung tâm

### Slide 15: Continual Learning - Giải pháp

**Định nghĩa:**
> Continual Learning = Khả năng học liên tục từ data mới **KHÔNG QUÊN** knowledge cũ

**3 Challenges:**
1. **Catastrophic Forgetting**: Model quên pattern cũ khi học data mới
2. **Data Heterogeneity**: Khác biệt distribution giữa sites
3. **Class Imbalance Evolution**: Tỷ lệ MCI thay đổi theo thời gian

### Slide 16: Continual Learning Approaches

**3 Strategies:**

**1. Regularization-based:**
- **Elastic Weight Consolidation (EWC)**:
  - Protect important weights khi update
  - Phạt những thay đổi lớn ở parameters quan trọng

**2. Rehearsal-based:**
- **Experience Replay**:
  - Lưu subset của old data
  - Mix với new data khi train
- **Privacy-preserving**: Use synthetic data (GAN)

**3. Architecture-based:**
- **Progressive Neural Networks**:
  - Thêm columns mới cho mỗi task
- **Dynamic Networks**:
  - Expand capacity khi cần

### Slide 17: Federated Learning + Continual Learning

**Integration Framework:**

```
Hospital A (Site 1) ────┐
  ├─ Local model        │
  ├─ Local data         │
  └─ Continual update   │
                        ├──→ Aggregation ──→ Global Model
Hospital B (Site 2) ────┤                      (Continual)
  ├─ Local model        │
  ├─ Local data         │
  └─ Continual update   │
                        │
Hospital C (Site 3) ────┘
  ├─ Local model
  ├─ Local data
  └─ Continual update
```

**Process:**
1. Each site trains **local continual model**
2. Share only **model updates** (not data!) → Privacy
3. Central server aggregates → **Global continual model**
4. Distribute back to sites
5. Sites continue learning **WITHOUT forgetting** global knowledge

**Key Benefits:**
- ✅ Privacy-preserving (data stays local)
- ✅ Adapt to new patterns continuously
- ✅ Retain knowledge from all sites
- ✅ Handle distribution shift across hospitals

### Slide 18: Application to Parkinson MCI Prediction

**Scenario:**
- **Site A**: Data từ 2010-2015 (old biomarker protocols)
- **Site B**: Data từ 2016-2020 (updated protocols)
- **Site C**: Data từ 2021-2025 (new imaging techniques)

**Without Continual Learning:**
- Retrain from scratch → Lose old knowledge
- Or keep static → Miss new patterns

**With Continual Learning:**
- Model learns from Site B **WITHOUT** forgetting Site A
- Model adapts to Site C **WHILE** retaining A+B knowledge
- **Result**: Best of all worlds!

### Slide 19: Research Questions for Future Work

**Open Questions:**
1. Which CL algorithm works best for medical federated settings?
   - EWC vs Replay vs Progressive Networks?

2. How to balance **stability** (retain old) vs **plasticity** (learn new)?
   - Optimal hyperparameters for medical data?

3. Privacy-preserving rehearsal methods?
   - Synthetic data generation without patient info?

4. Benchmark datasets?
   - Need multi-center Parkinson datasets with temporal evolution

**Proposed Experiments:**
- Test CL algorithms on PPMI multi-site data
- Compare FL+CL vs traditional centralized training
- Measure catastrophic forgetting in medical models

---

## **PHẦN 6: KẾT LUẬN** (2-3 phút)

### Slide 20: Key Takeaways

**1. Model Selection:**
- Advanced models (Boosting, Deep Learning) often best
- BUT: Simple models + good preprocessing can compete!
- Paper Parkinson: Ensemble (AUC 0.80-0.83)
- My Experiment: Decision Tree + SMOTE-ENN (PR-AUC 0.8023)

**2. Preprocessing Matters:**
- **Imbalance handling**: Critical for medical data
- **Feature selection**: Balance performance vs interpretability
- **Scaling**: Depends on model type (not needed for trees)

**3. Future Direction - Continual Learning:**
- Essential for **multi-center medical AI**
- Federated Learning + Continual Learning = **Scalable + Private**
- Research opportunities in:
  - Anti-forgetting techniques
  - Privacy-preserving methods
  - Multi-site validation

### Slide 21: References & Contact
- Paper: "Predicting Cognitive Decline in Parkinson's Disease" (Frontiers in Neuroscience, 2023)
- My Experiment Code: [GitHub Link]
- Contact: [Email]

**Thank you! Questions?**

---

## **PHỤ LỤC: BACKUP SLIDES**

### Backup 1: Detailed Preprocessing Pipeline
- Step-by-step preprocessing trong experiment
- Code snippets

### Backup 2: Statistical Tests
- Wilcoxon signed-rank test for model comparison
- Cross-validation strategy

### Backup 3: Feature Importance Analysis
- Top features from best models
- SHAP values visualization

### Backup 4: Error Analysis
- Confusion matrices
- Failure cases analysis

---

## **TIMING BREAKDOWN (TOTAL: 25-30 phút)**

| Phần | Thời gian | Slides |
|------|-----------|--------|
| 1. Giới thiệu | 2-3 min | 1-2 |
| 2. Paper Methods | 5-7 min | 3-5 |
| 3. **Thực nghiệm** | **8-10 min** | **6-11** |
| 4. Paper Results | 3-5 min | 12-13 |
| 5. **Continual Learning** | **5-7 min** | **14-19** |
| 6. Kết luận | 2-3 min | 20-21 |
| Q&A | 5 min | - |

**Total**: 25-30 phút + Q&A

---

## **NOTES CHO THUYẾT TRÌNH**

### Điểm nhấn:
1. **Slide 8-10**: Thực nghiệm của bạn → Show expertise
2. **Slide 16-18**: Continual Learning → Innovation/Future work
3. Sử dụng 4 biểu đồ từ `experiments/presentation_plots/`

### Tips:
- Lướt nhanh phần paper background (Slide 3-5)
- Dành thời gian cho Slide 8-11 (thực nghiệm)
- **Emphasize** Slide 16-19 (Continual Learning - định hướng chính)
- Prepare backup slides cho Q&A technical

### Câu hỏi dự kiến:
1. "Why not use deep learning?" → Slide 8 (dataset size limitation)
2. "How to prevent catastrophic forgetting?" → Slide 16 (EWC, Replay)
3. "Privacy concerns?" → Slide 17 (Federated Learning)
4. "Practical deployment?" → Slide 11 (Performance vs Time trade-off)
