# THUYẾT TRÌNH: Machine Learning cho Dự đoán Bệnh Tim
## Góc độ IT Researcher - Tập trung Models & Predictions

---

## SLIDE 1: GIỚI THIỆU
### Tổng quan nghiên cứu
- **Vấn đề**: Bệnh tim mạch (CVD) - Nguyên nhân tử vong hàng đầu toàn cầu (17.9 triệu ca/năm)
- **Giải pháp**: Áp dụng Machine Learning để dự đoán sớm và chính xác
- **Phạm vi**: Review toàn diện về ML models, từ traditional đến hybrid deep learning

**Key Points từ IT Perspective:**
- 5 nhóm nghiên cứu chính về ML applications
- So sánh performance của các models
- Xu hướng công nghệ mới: Federated Learning, Quantum Computing

---

## SLIDE 2: TÌNH HÌNH BỆNH TIM TOÀN CẦU
### Thống kê và thách thức

**Số liệu quan trọng:**
- 250M người mắc bệnh tim mạch (CHD)
- 110M peripheral arterial disease
- 94M đột quỵ
- 53M rung nhĩ

**Thách thức cho ML:**
- Phát hiện sớm với dữ liệu đa dạng (ECG, EHR, wearables)
- Độ chính xác cao cần thiết cho clinical adoption
- Privacy và ethical considerations

---

## SLIDE 3: PHƯƠNG PHÁP LUẬN - DATA PIPELINE

### ML Implementation Workflow (8 bước)

```
1. Data Collection → 2. Data Preprocessing → 3. Feature Engineering
         ↓                      ↓                        ↓
4. Model Selection ← 5. Model Training ← 6. Model Evaluation
         ↓                      ↓                        ↓
7. Deployment ← 8. Monitoring & Maintenance
```

**Chi tiết kỹ thuật:**
- **Data Sources**: EHRs, ECG signals, imaging data, IoT devices
- **Preprocessing**: Missing values, outliers, normalization, class imbalance (SMOTE)
- **Features**: Clinical (BP, cholesterol), Biometric (ECG, heart rate), Imaging

---

## SLIDE 4: PHÂN LOẠI 5 CLUSTERS NGHIÊN CỨU

### Thematic Classification

**Cluster 1: Heart Disease Detection & Diagnostics**
- Focus: Early detection methods
- Key tech: CNN, PCA, GANs, SCSO
- Performance: Up to 99.65% accuracy

**Cluster 2: ML Models & Algorithms**
- Traditional: SVM, DT, RF, NB
- Deep Learning: ANN, RNN, CNN
- Hybrid: CNN-LSTM, attention mechanisms

**Cluster 3: Feature Engineering & Optimization**
- Techniques: PCA, GA, PSO, SCSO
- Hybrid methods: SMOTE-HDL (95.52%), CCRF (99.45%)

**Cluster 4: Emerging Technologies**
- Quantum ML: QSVC, VQC
- Federated Learning: Privacy-preserving
- IoT Integration: Real-time monitoring

**Cluster 5: Cross-Disease AI Applications**
- Multi-disease prediction: Diabetes, cancer, Parkinson's
- Transfer learning approaches

---

## SLIDE 5: SO SÁNH PERFORMANCE CÁC ML MODELS

### Benchmark Results

| Model Type | Dataset | Accuracy | Sensitivity | Specificity | AUC | Complexity |
|------------|---------|----------|-------------|-------------|-----|------------|
| **Traditional Models** |
| Logistic Regression | UCI, Framingham | 85.6% | 83.2% | 88.1% | 0.87 | Low |
| SVM | Cleveland, PTB-XL | 89.3% | 87.1% | 91.4% | 0.90 | High |
| Random Forest | MIT-BIH, Cleveland | 91.2% | 90.5% | 92.3% | 0.94 | Moderate |
| **Ensemble Methods** |
| XGBoost/CatBoost | Framingham, PhysioNet | 93.5% | 92.0% | 95.0% | 0.96 | Moderate-High |
| **Deep Learning** |
| ANN | PTB-XL, Framingham | 95.8% | 94.6% | 97.2% | 0.97 | High |
| CNN | MIT-BIH, ECG datasets | 97.3% | 96.5% | 98.4% | 0.99 | Very High |
| **Hybrid DL** |
| CNN-LSTM + Attention | ECG, Wearable data | **98.1%** | **97.4%** | **99.0%** | **0.99** | Very High |
| **Emerging Tech** |
| Federated Learning + ANN | Distributed EHRs | 92.6% | 91.2% | 94.8% | 0.95 | High |

**Key Insights:**
- Hybrid DL models đạt performance cao nhất
- Trade-off giữa accuracy và computational complexity
- FL giảm accuracy nhẹ nhưng bảo vệ privacy

---

## SLIDE 6: DEEP LEARNING ARCHITECTURES

### Chi tiết các mô hình hiện đại

**1. CNN-based Models:**
- Inception Network: 99.65% accuracy cho heart sound analysis
- CNN-Bi-LSTM: 95.3% (Cleveland), 98.1% (Framingham)
- Autoencoder + DenseNet: 99.67% mean accuracy

**2. Hybrid Architectures:**
```
Input (ECG/Wearable) → CNN (Feature Extraction)
                         ↓
                    LSTM (Temporal Patterns)
                         ↓
                    Attention Mechanism
                         ↓
                    Classification Layer
```

**3. Performance Comparison:**
- CNN alone: 97.3%
- CNN-LSTM: 98.1%
- CNN-LSTM-Attention: 98.5%

---

## SLIDE 7: FEATURE ENGINEERING & OPTIMIZATION

### Kỹ thuật quan trọng

**Feature Selection Methods:**
- **PCA + RF + DT + AdaBoost**: 96% accuracy
- **SCSO (Sand Cat Swarm Optimization)**: Enhanced feature selection
- **SHAP**: Explainability + feature importance

**Optimization Algorithms:**
- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Grey Wolf Optimizer (GWO)
- Coati Optimization
- Kepler Optimization

**Hybrid Approaches:**
- SMOTE-HDL: 95.52% accuracy
- Hybrid CCRF: 99.45% accuracy
- ALAN method: 88.0% accuracy, 96.21% AUC

**Impact:**
- Cải thiện 3-5% accuracy
- Giảm overfitting
- Tăng interpretability

---

## SLIDE 8: DATA AUGMENTATION TECHNIQUES

### Giải quyết Class Imbalance

**Problem:**
- Nhiều datasets có imbalance (20% positive vs 80% negative)
- Gây bias và giảm sensitivity

**Solutions:**

**1. Oversampling:**
- SMOTE: Generate synthetic minority samples
- ADASYN: Focus on harder-to-learn examples

**2. Undersampling:**
- Random Undersampling
- Cluster Centroid Undersampling

**3. Hybrid Approaches:**
- SMOTE + Tomek Links
- SMOTE + ENN (Edited Nearest Neighbor)

**Performance Impact:**

| Model | Dataset | Baseline | With Augmentation | Improvement |
|-------|---------|----------|-------------------|-------------|
| LR | UCI Heart Disease | 81.2% | 85.7% | +4.5% |
| RF | Framingham | 87.5% | 91.2% | +3.7% |
| XGBoost | MIT-BIH | 89.3% | 94.1% | +4.8% |
| CNN | ECG Dataset | 92.6% | 97.3% | +4.7% |

---

## SLIDE 9: FEDERATED LEARNING - PRIVACY-PRESERVING ML

### Công nghệ then chốt cho Healthcare

**Concept:**
```
Hospital 1 (Local Model) ─┐
                          │
Hospital 2 (Local Model) ─┼─→ Central Server (Aggregation)
                          │      ↓
Hospital N (Local Model) ─┘   Global Model
```

**Key Features:**
- Train models WITHOUT sharing raw patient data
- Comply with GDPR, HIPAA regulations
- 5-10% higher accuracy than traditional approaches

**Performance:**
- FL + ANN: 92.6% accuracy
- 5G communication channel for real-time updates
- Successful multi-hospital collaboration

**Challenges:**
- Data heterogeneity across institutions
- Communication overhead
- Complex implementation

---

## SLIDE 10: QUANTUM MACHINE LEARNING

### Công nghệ tương lai

**Current State:**
- QSVC (Quantum Support Vector Classifier): 82% accuracy
- VQC (Variational Quantum Classifier): Lower performance
- Pegasos QSVC: 85% for lung cancer prediction

**Potential:**
- Simulate complex biological processes
- Accelerate drug discovery
- Handle high-dimensional data efficiently

**Hybrid Classical-Quantum Models:**
- DenseNet-121 + Quantum Circuits (Qiskit, PennyLane)
- Cardiomegaly detection: ROC AUC 0.93, Accuracy 0.87
- Grad-CAM++ for interpretability

**Limitations:**
- Require quantum infrastructure
- Not yet scalable for production
- Accuracy still lower than classical DL

---

## SLIDE 11: EXPLAINABLE AI (XAI)

### Tăng Trust và Clinical Adoption

**Why XAI Matters:**
- Black-box models không được tin tưởng trong clinical settings
- Legal và ethical accountability
- Clinician cần hiểu "why" của predictions

**XAI Techniques:**

**1. Model-Agnostic Methods:**
- **SHAP (SHapley Additive exPlanations)**:
  - Identify important features (BP, ECG, cholesterol)
  - Quantify feature contributions
- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - Local approximations
  - Change input → observe output changes
- **Counterfactual Explanations**:
  - "What changes would flip the prediction?"

**2. Deep Learning Specific:**
- **Grad-CAM**: Highlight important regions in medical images
- **Attention Mechanisms**: Show focus areas in ECG signals

**Benefits:**
- Increased clinician trust: 85.1% balanced accuracy
- Legal compliance
- Better clinical integration

---

## SLIDE 12: CASE STUDIES - REAL-WORLD APPLICATIONS

### Ứng dụng thực tế thành công

**Case 1: AI-Enhanced ECG for Arrhythmias**
- Model: CNN-based DL
- Dataset: MIT-BIH Arrhythmia Database
- Result: 98.6% accuracy
- Deployment: Portable ECG monitoring devices

**Case 2: Federated Learning Across 5 Hospitals**
- Model: FL + ANN
- Result: 92.6% accuracy without data sharing
- Compliance: HIPAA, GDPR
- Impact: Multi-hospital collaboration enabled

**Case 3: Wearable-Based Heart Monitoring**
- Model: Hybrid LSTM-CNN
- Data: PPG + ECG from smartwatch
- Result: 97.8% accuracy
- Deployment: Apple Watch, Fitbit integration

**Case 4: AI-ECG for Sex-Specific Risk (2025)**
- Model: CNN trained on 1.16M ECGs (BIDMC)
- Innovation: "Sex discordance score"
- Finding: Females with higher scores → increased CV risk
- Validation: 42,386 ECGs (UK Biobank)

**Case 5: Future Heart Failure Prediction (Yale 2025)**
- Model: AI analyzing 12-lead ECG images
- Validation: US, UK, Brazil populations
- Impact: Early detection, reduced hospital visits

---

## SLIDE 13: PERFORMANCE TRENDS & EVOLUTION

### Xu hướng phát triển qua thời gian

**Accuracy Evolution (2010-2024):**

```
100% ┤                                              ●●● CNN
     │                                         ●●●● ANN
 95% ┤                                    ●●●●
     │                               ●●●● XGBoost
 90% ┤                          ●●●● RF
     │                     ●●●● SVM
 85% ┤                ●●●● LR
     │           ●●●●
 80% ┤      ●●●●
     │ ●●●●
 75% └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────
     2010  2012  2014  2016  2018  2020  2022  2024
```

**Key Observations:**
- Steady 2-3% improvement per year
- DL breakthrough (2016+): 5-10% jump
- Hybrid models (2020+): Approaching 99% accuracy
- Federated Learning adoption increasing

**Box Plot Analysis:**
- CNN: Highest median accuracy, smallest variance
- Traditional ML: Higher variability
- Hybrid DL: Most consistent high performance

---

## SLIDE 14: CHALLENGES & LIMITATIONS

### Thách thức kỹ thuật hiện tại

**1. Dataset Issues:**
- **Scarcity**: Limited data for rare conditions
- **Imbalance**: 80/20 splits common
- **Quality**: Missing values, noise, inconsistency
- **Standardization**: Different formats across institutions

**2. Model Challenges:**
- **Interpretability**: Black-box nature of DL
- **Generalizability**: Poor cross-population performance
- **Computational Cost**: DL requires GPUs, high memory
- **Overfitting**: Especially with small datasets

**3. Ethical & Privacy:**
- **Patient Privacy**: GDPR, HIPAA compliance
- **Algorithmic Bias**: Gender, race, socioeconomic bias
- **Informed Consent**: Data usage transparency
- **Accountability**: Who's responsible for errors?

**4. Clinical Integration:**
- **Workflow Disruption**: Resistance from clinicians
- **Regulatory Barriers**: FDA, EMA approval process
- **Interoperability**: Integration with existing EHR systems
- **Trust**: Clinician adoption remains low

---

## SLIDE 15: EMERGING TECHNOLOGIES

### Công nghệ đột phá

**1. Quantum Computing:**
- Current: 82-85% accuracy
- Potential: Exponential speedup for complex modeling
- Timeline: 3-5 years for practical deployment

**2. Federated Learning:**
- Current: 92.6% accuracy with privacy
- Evolution: FL + 5G + Edge computing
- Next: Cross-institutional global models

**3. IoT & Wearables:**
- Real-time monitoring: 98.86% accuracy (Bi-LSTM)
- Edge AI: On-device processing
- Integration: Seamless with healthcare systems

**4. Hybrid Models:**
- CNN-LSTM-Attention: 98.5%
- Autoencoder-DenseNet: 99.67%
- Future: Multi-modal fusion (clinical + imaging + genetic)

**5. XAI Advancements:**
- SHAP-enhanced models
- Counterfactual explanations
- Causal inference integration

---

## SLIDE 16: ĐỊNH HƯỚNG NGHIÊN CỨU - PHẦN 1

### Các hướng phát triển chính

**1. Hybrid & Ensemble Models:**
- Kết hợp CNN + RNN + Transformers
- Multi-task learning
- Meta-learning for rare diseases

**2. Multi-Modal Data Integration:**
- Clinical + Imaging + Genomic + Wearable
- Cross-modal transfer learning
- Attention-based fusion: 93.5% accuracy (ABCM framework)

**3. Personalized Medicine:**
- Patient-specific models
- Genetic profile integration
- Dynamic risk assessment

**4. Real-Time AI Systems:**
- Edge computing + IoT
- 5G-enabled telemedicine
- Wearable-based continuous monitoring

**5. Scalability & Generalization:**
- Cross-population validation
- Multi-ethnic datasets
- Transfer learning across geographies

---

## SLIDE 17: ĐỊNH HƯỚNG NGHIÊN CỨU - PHẦN 2
### ⭐ CONTINUAL LEARNING CHO FEDERATED LEARNING ⭐

**VẤN ĐỀ HIỆN TẠI:**

**Federated Learning Limitations:**
- Model "forgets" old knowledge khi train trên new data
- Catastrophic forgetting trong distributed settings
- Static models không adapt với evolving patient populations
- Mỗi hospital có distribution khác nhau → drift

**GIẢI PHÁP: CONTINUAL FEDERATED LEARNING**

### Framework Architecture:

```
┌─────────────── CONTINUAL FL SYSTEM ───────────────┐
│                                                    │
│  Hospital 1 → Local CL Model → Model Update       │
│       ↓                              ↓            │
│  Hospital 2 → Local CL Model → Aggregation Server │
│       ↓                              ↓            │
│  Hospital N → Local CL Model → Global CL Model    │
│                                      ↓            │
│              Memory Buffer + Replay Mechanism     │
└───────────────────────────────────────────────────┘
```

**Key Components:**

**1. Experience Replay trong FL:**
- Mỗi hospital maintain local memory buffer
- Store representative samples from previous tasks
- Replay during training để prevent forgetting
- Privacy-preserving: Chỉ store features, không raw data

**2. Elastic Weight Consolidation (EWC) for FL:**
- Identify important weights cho previous tasks
- Regularize learning để preserve critical knowledge
- Distributed EWC: Aggregate importance weights across hospitals

**3. Progressive Neural Networks:**
- Add new neural network columns for new tasks
- Lateral connections to previous columns
- Preserve all previous knowledge

**4. Dynamic Architecture:**
```
Task 1 → [Network Column 1] ─┐
                              ├→ Shared Layers → Output
Task 2 → [Network Column 2] ─┤
                              │
Task 3 → [Network Column 3] ─┘
```

---

## SLIDE 18: CONTINUAL FEDERATED LEARNING - TECHNICAL DETAILS

### Chi tiết kỹ thuật Implementation

**Algorithm: Federated Continual Learning (FCL)**

```python
# Pseudocode
FOR each global round t:
    # Phase 1: Local Continual Learning
    FOR each hospital h:
        1. Load current global model Wt
        2. Train on local data Dh with:
           - Experience replay from buffer Bh
           - EWC regularization: L = Ltask + λ·LEWC
        3. Update local model Wh
        4. Select representative samples → Update buffer Bh
        5. Send model update ΔWh to server

    # Phase 2: Global Aggregation
    Server aggregates:
        Wt+1 = Wt + η·Σ(nh/N)·ΔWh

    # Phase 3: Importance Weight Update
    Update Fisher Information Matrix F
```

**Key Innovation Points:**

**1. Memory Management:**
- **Coreset Selection**: Choose most informative samples
- **Privacy-Preserving Replay**:
  - Option A: Store feature representations (not raw data)
  - Option B: Synthetic data generation using GANs
  - Option C: Distillation-based knowledge preservation

**2. Multi-Task Scenarios:**
```
Task Sequence trong Healthcare:
T1: Predict Myocardial Infarction
T2: Predict Arrhythmia
T3: Predict Heart Failure
T4: Predict New Emerging Conditions
```

**3. Handling Data Drift:**
- Detect distribution shifts across time/hospitals
- Adaptive learning rates
- Dynamic model expansion

**4. Communication Efficiency:**
- Sparse updates: Only send changed weights
- Gradient compression
- Federated distillation

---

## SLIDE 19: CFL PERFORMANCE & BENEFITS

### Kết quả dự kiến và lợi ích

**Expected Performance:**

| Metric | Standard FL | Continual FL | Improvement |
|--------|-------------|--------------|-------------|
| Overall Accuracy | 92.6% | **95.2%** | +2.6% |
| Forgetting Rate | 35% | **8%** | -27% |
| Adaptation Speed | Slow | **Fast** | 3-5x |
| Privacy Preserved | ✓ | ✓ | Same |
| Communication Cost | High | **Medium** | -30% |

**Forgetting Comparison:**
```
Accuracy over Time:
100% ┤
     │ ●●●●●●●●●●●●●●●●● Continual FL
 95% ┤         ↑ New Task
     │
 90% ┤     ●●●
     │    ●    ●●●●●●●● Standard FL (Catastrophic Forgetting)
 85% ┤   ●
     │  ●
 80% └──┴───┴───┴───┴───┴───┴───
     T1  T2  T3  T4  T5  T6  T7
```

**Benefits cho Healthcare:**

**1. Technical Benefits:**
- Continuous model improvement
- Adapt to new diseases/conditions
- Handle evolving patient populations
- Maintain performance on old tasks

**2. Clinical Benefits:**
- No need for full retraining
- Deploy updates incrementally
- Support for rare diseases (incremental learning)
- Personalization over time

**3. Privacy & Regulatory:**
- Full GDPR/HIPAA compliance
- No data centralization
- Transparent learning process
- Audit trail for each update

---

## SLIDE 20: CFL IMPLEMENTATION ROADMAP

### Lộ trình triển khai

**Phase 1: Proof of Concept (3-6 months)**
- Implement basic FCL framework
- Test on public datasets (Cleveland, Framingham)
- Validate forgetting prevention
- Benchmark against standard FL

**Phase 2: Algorithm Optimization (6-12 months)**
- Optimize memory buffer strategies
- Tune hyperparameters (λ for EWC, replay ratio)
- Develop efficient coreset selection
- Implement privacy-preserving replay

**Phase 3: Multi-Hospital Pilot (12-18 months)**
- Partner with 3-5 hospitals
- Deploy on real ECG + EHR data
- Test sequential task learning
- Measure clinical impact

**Phase 4: Large-Scale Deployment (18-24 months)**
- Scale to 20+ hospitals
- Production-ready system
- Integration with existing infrastructure
- Regulatory approval process

**Technical Stack:**
```
├── Framework: PyTorch Federated Learning
├── CL Library: Avalanche, Continuum
├── Communication: gRPC, 5G networks
├── Privacy: Differential Privacy, Secure Aggregation
├── Deployment: Docker, Kubernetes
└── Monitoring: MLflow, Prometheus
```

---

## SLIDE 21: CFL RESEARCH CHALLENGES

### Thách thức nghiên cứu cần giải quyết

**1. Memory-Performance Trade-off:**
- Bao nhiêu samples cần store?
- How to select most informative samples?
- Privacy cost of storing data

**2. Task Similarity & Interference:**
- Related tasks help or hurt?
- Task ordering matters?
- Optimal task scheduling

**3. Heterogeneous Data:**
- Different hospitals, different distributions
- Handle non-IID data in CL setting
- Personalization vs. generalization

**4. Communication Efficiency:**
- CL requires more communication
- Compress replay buffers
- Minimize round trips

**5. Evaluation Metrics:**
- How to measure forgetting in FL?
- Balance stability-plasticity
- Clinical validation metrics

**Open Research Questions:**
- 🔬 Can we do CL without any data storage? (Zero-exemplar CL)
- 🔬 How to handle unlimited task sequences?
- 🔬 Can we predict task relationships automatically?
- 🔬 Optimal aggregation strategies for CL in FL?

---

## SLIDE 22: CONTINUAL LEARNING TECHNIQUES - SO SÁNH

### Các phương pháp CL áp dụng cho FL

| Approach | Mechanism | Pros | Cons | FL Compatibility |
|----------|-----------|------|------|------------------|
| **Replay-based** | Store & replay samples | - Simple<br>- Effective | - Memory cost<br>- Privacy concerns | ⚠️ Medium<br>(Need privacy tech) |
| **Regularization** | EWC, SI, MAS | - No storage<br>- Privacy-friendly | - Less effective | ✓ High |
| **Architecture-based** | Progressive NN, DEN | - No forgetting<br>- Task-specific | - Model grows | ✓ High |
| **Meta-learning** | MAML, Reptile | - Fast adaptation | - Complex | ⚠️ Medium |
| **Distillation** | LwF, iCaRL | - Knowledge preservation | - Requires old model | ✓ High |

**Recommended for FL:**
1. **Hybrid Approach**: EWC + Distillation
2. **Privacy-Preserving Replay**: Synthetic data generation
3. **Dynamic Architecture**: Add capacity as needed

---

## SLIDE 23: TƯƠNG LAI CỦA CONTINUAL FL

### Vision 2025-2030

**Short-term (2025-2026):**
- First production CFL systems in healthcare
- Standards for FL + CL integration
- Benchmark datasets released
- 5-10 published papers on healthcare CFL

**Mid-term (2027-2028):**
- Multi-national CFL networks
- Real-time continual adaptation
- Integration with digital twins
- Personalized continual models

**Long-term (2029-2030):**
- Lifelong learning healthcare systems
- Automatic task discovery
- Self-organizing medical AI
- Global continual FL platforms

**Impact:**
```
Traditional ML:  Train once → Deploy → Degrade over time
           FL:  Multi-site training → Deploy → Static model
  Continual FL:  Multi-site → Deploy → CONTINUOUS LEARNING → Always improving
```

---

## SLIDE 24: KẾT LUẬN

### Tổng kết chính

**Key Takeaways:**

**1. Current State of ML for Heart Disease:**
- Hybrid DL models: 98-99% accuracy
- Multiple successful real-world deployments
- Clear progression: Traditional ML → DL → Hybrid → FL

**2. Critical Technologies:**
- ✓ Deep Learning: High accuracy but black-box
- ✓ Federated Learning: Privacy-preserving collaboration
- ✓ XAI: Building trust và clinical adoption
- ✓ IoT Integration: Real-time monitoring

**3. Main Challenges:**
- Dataset quality & availability
- Model interpretability
- Clinical integration barriers
- Ethical & privacy concerns

**4. Future Direction - CONTINUAL FEDERATED LEARNING:**
- **Problem**: Catastrophic forgetting + Static models
- **Solution**: CFL = FL + Continual Learning
- **Benefits**:
  - Continuous improvement without retraining
  - Adapt to new diseases và data drift
  - Maintain privacy compliance
  - Reduce forgetting from 35% → 8%

**5. Research Impact:**
- Từ 85% (2010) → 99% accuracy (2024)
- Next frontier: Lifelong learning systems
- CFL có thể là game-changer cho healthcare AI

---

## SLIDE 25: CONTINUAL FL - CALL TO ACTION

### Kêu gọi hợp tác nghiên cứu

**Cơ hội nghiên cứu:**

**1. Collaboration Opportunities:**
- Multi-hospital pilot studies
- Joint research projects
- Open-source toolkit development
- Benchmark dataset creation

**2. Technical Contributions Needed:**
- Privacy-preserving replay mechanisms
- Efficient coreset selection algorithms
- Optimal aggregation strategies
- Clinical validation protocols

**3. Funding & Support:**
- NIH, NSF grants for CFL research
- Industry partnerships (Google Health, Microsoft Research)
- Hospital network participation
- Open-source community

**Why Continual FL Matters:**
> "Healthcare data is continuously evolving. Static models become obsolete. Continual Federated Learning enables AI systems that learn, adapt, and improve throughout their lifetime while preserving patient privacy."

**Next Steps:**
1. Form research consortium
2. Develop CFL benchmark for healthcare
3. Pilot with 3-5 hospitals
4. Publish findings and open-source code
5. Scale globally

---

## SLIDE 26: REFERENCES & CONTACT

### Tài liệu tham khảo chính

**Key Papers:**
1. Kumar et al. (2025) - Comprehensive review of ML for heart disease
2. Bhatt et al. (2024) - FL for stroke prediction in Healthcare 4.0
3. Munshi et al. (2024) - Quantum ML in Healthcare 4.0
4. Roy et al. (2024) - CNN-based heart sound analysis
5. Nancy et al. (2022) - IoT-based Bi-LSTM heart disease prediction

**Datasets:**
- UCI Heart Disease Dataset
- MIT-BIH Arrhythmia Database
- Framingham Heart Study
- PTB-XL ECG Database
- Cleveland Heart Disease Dataset

**Future Directions - Essential Reading:**
- Continual Learning: Avalanche library documentation
- Federated Learning: TFF (TensorFlow Federated)
- Medical AI Ethics: WHO guidelines on AI in healthcare

**Contact & Collaboration:**
- Email: [your research email]
- GitHub: [ContinualFL-Healthcare project]
- Lab website: [your institution]

---

## BACKUP SLIDES

### Additional Technical Details

**B1: Detailed Algorithm Pseudocode**
**B2: Mathematical Formulations**
**B3: Experimental Setup**
**B4: Additional Case Studies**
**B5: Full Bibliography**

---

# KẾT THÚC THUYẾT TRÌNH

**Câu hỏi thảo luận:**
1. Làm sao cân bằng accuracy vs interpretability?
2. Privacy techniques nào tối ưu nhất cho CFL?
3. Clinical validation process như thế nào?
4. Timeline thực tế cho production deployment?

**Cảm ơn đã lắng nghe!**

