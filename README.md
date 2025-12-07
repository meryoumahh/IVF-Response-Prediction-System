# IVF Response Prediction System

ML system predicting ovarian stimulation response in IVF using clinical data and PDF extraction.

---

## 🎯 Problem Statement

Predict IVF response categories:
- **Low Response**
- **Optimal Response**
- **High Response**

---

## 📊 Data Loading & Preprocessing

### 1. Data Sources

**Historical Data**: `data/raw/patients.csv` (254 patient records)

**PDF Reports**: `data/raw/pdf_reports/` (medical reports in PDF format)

### 2. PDF Extraction

**Module**: `src/preprocessing/pdf_extractor.py`

Extracts 9 clinical parameters using regex patterns:
- Demographics: Name, Birth Date, Age
- Biomarkers: AMH, AFC, n_Follicles
- Treatment: Protocol, Cycle Number
- Outcome: Response, E2_day5

**E2_day5 Special Handling**:
```python
def extract_j5_e2(text):
    j_matches = list(re.finditer(r"J", text, re.IGNORECASE))
    sixth_j = j_matches[5]  # Day 5 marker in monitoring table
    content_after_6th_j = text[sixth_j.end():]
    target_pattern = r"\d{1,2}/\d{1,2}.*?\s(\d{3,4})\s"
    match = re.search(target_pattern, content_after_6th_j)
    return float(match.group(1)) if match else None
```

### 3. Missing Data Imputation

Evidence-based strategy (lowest → highest missing %):

| Feature | Method | Rationale |
|---------|--------|-----------|
| **Age** | Median (32y) | Narrow distribution (25-40y) |
| **n_Follicles** | Age-stratified | ≤34→10, 35-39→20, ≥40→30 |
| **AMH** | Linear regression | Strong correlation with n_Follicles (r=0.68) |
| **AFC** | Age + n_Follicles | Must align with actual follicle count |
| **E2_day5** | Follicle-based median | More follicles → more E2 production |

**Why this order?** Later imputations use earlier imputed features, reducing error propagation.

### 4. Patient Anonymization

**Experience 1**: Multi-cycle tracking (same patient = same ID)
- Output: `clean_dataset_experience1.csv`

**Experience 2**: Independent rows (each row = new patient)
- Output: `clean_dataset_experience2.csv`
- **Used for model training** (maximum samples)

---

## 🔍 Exploratory Data Analysis (EDA)

**Notebook**: `notebooks/visualizations.ipynb`

### Key Findings

**1. Feature Correlations**:
- AMH ↔ n_Follicles: r=0.68 (strong positive)
- Age ↔ AMH: r=-0.42 (moderate negative)
- AFC ↔ n_Follicles: r=0.55 (moderate positive)

**2. Response Distribution**:
- Low Response: 35% of patients
- Optimal Response: 42% of patients
- High Response: 23% of patients

**3. Age-Stratified Patterns**:
```
Age Group    Avg Follicles    Avg AMH
≤34          21.5             3.2
35-39        15.8             2.1
≥40          9.3              1.4
```

**4. Follicle Group Analysis**:
```
Group          Age    AMH    n_Follicles    E2_day5
Low (≤18)      32.2   1.5    9.3            1174
Medium (19-24) 31.1   3.0    21.6           1334
High (>24)     30.8   4.4    32.2           1072
```

**Insights**:
- Younger patients tend to have higher AMH and follicle counts
- E2 levels don't linearly increase with follicles (high responders may have lower E2)
- AMH is the strongest predictor of response

---

## 🤖 Model Selection & Training

### Algorithm: Random Forest Classifier

**Why Random Forest?**
- Handles non-linear medical relationships(see plots in visualizations.ipynb)
- Robust to outliers in small datasets
- Built-in feature importance
- No feature scaling required
- Handles class imbalance with `class_weight='balanced'`

### Configuration
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
```

### Training Strategy

**1. Stratified Train-Test Split** (80/20)
- Maintains class distribution in both sets

**2. 5-Fold Cross-Validation**
- Stratified sampling
- Provides confidence intervals
- Detects overfitting

**3. Feature Engineering**
- One-hot encoding for Protocol
- Binary encoding for categorical features

### Performance Metrics

**Cross-Validation (5-Fold)**:
- Mean Accuracy: **81.6% ± 3.8%**
- Folds: [76.5%, 81.5%, 83.8%, 78.7%, 87.5%]

**Test Set**:
- Accuracy: **92.1%**
- Precision: 91-93% across classes
- Recall: 88-97% across classes


### Feature Importance

| Feature | Importance | Clinical Significance |
|---------|-----------|----------------------|
| AMH | 32.4% | Primary ovarian reserve marker |
| AFC | 24.3% | Direct follicle measurement |
| n_Follicles | 21.5% | Historical response |
| E2_day5 | 9.0% | Hormone level indicator |
| Age | 7.8% | Patient age factor |
| Protocol | 4.5% | Treatment type |

**Validation**: AMH and AFC as top predictors aligns with medical literature.

### Explainability (SHAP)

**Module**: SHAP TreeExplainer

- Provides per-prediction explanations
- Shows feature contribution for each patient
- Builds clinical trust
- Outputs: `shap_summary_plot.png`, `shap_beeswarm_high.png`

---

## 🖥️ Web Application

**Module**: `src/app/app.py` (Streamlit)

### Features

**Input Form**:
- Age (20-45)
- AMH Level (0-20)
- AFC (0-50)
- Cycle Number (1-5)
- Protocol (Flexible/Fixed Antagonist, Agonist)

**Output**:
- Predicted response category
- Confidence scores for all classes
- Probability distribution chart

**Example Predictions**:

```
Patient 1: Age=40, AMH=0.8, AFC=5
→ Low Response (85% confidence)

Patient 2: Age=28, AMH=5.5, AFC=38
→ High Response (92% confidence) - OHSS risk

Patient 3: Age=32, AMH=3.2, AFC=20
→ Optimal Response (88% confidence)
```

---

## 🚀 How to Run

### Installation

```bash
cd TanitAiTest
pip install -r requirements.txt
```

### Train Model

```bash
cd src/model
python random_forest_classifier.py
```

**Output**:
- Model saved to `models/ivf_prediction_model.pkl`
- Plots saved to `outputs/plots/`
- Console shows CV scores and test accuracy

### Launch Application

```bash
cd src/app
streamlit run app.py
```

**Access**: http://localhost:8501

---

## 📈 Key Insights & Clinical Applications

### Model Insights

1. **AMH is strongest predictor** (32.4%)
   - Validates medical literature
   - Model learns real clinical patterns

2. **Age matters but isn't deterministic** (7.8%)
   - Other factors (AMH, AFC) can compensate
   - Younger ≠ always better response

3. **Protocol has minimal impact** (4.5%)
   - Patient characteristics > treatment type
   - Suggests good protocol selection by clinicians

4. **High recall for Low Response** (97%)
   - Critical for identifying patients needing protocol adjustments

### Clinical Applications

**1. Treatment Planning**
- Predict response before stimulation
- Optimize protocol selection
- Adjust medication dosing



**2. Patient Counseling**
- Provide evidence-based expectations
- Improve informed consent
- Manage patient anxiety




---

## 🔮 Future Improvements

- Fix data leakage (train-only imputation)
- Hyperparameter tuning (grid search)
- Ensemble methods (RF + XGBoost)
- Time-series features for multi-cycle patients
- External validation (different clinics)
- EMR system integration
- Docker deployment
- Using OCR system to extract data from different input types

---

## 📚 References

**Clinical**:
- AFC: https://cyprusivfhospital.com/fr/quest-ce-quun-comptage-de-follicules-antraux-afc/
- E2: https://ada.com/hormones/estradiol/
- Guidelines: https://drive.google.com/file/d/15yKwgHvbGq4hc11vDiz_ETOSgysQ0Vjf/view

**Technical**: Random Forest (Breiman 2001) • SHAP (Lundberg 2017) • Cross-Validation (Kohavi 1995)

**Tools**: pdfplumber • scikit-learn • SHAP • Streamlit • pandas

---

