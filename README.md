# 🧹 Chore Bundling Intelligence Dashboard

**Business Question:**  
*What should service packages contain — which chores do students want bundled together, so you're not selling à la carte when a "Starter Pack" would convert better?*

---

## 📁 Project Structure

```
.
├── chore_bundling_dashboard.py     # Main Streamlit app
├── requirements.txt                # Python dependencies (pinned)
├── README.md                       # This file
└── data/
    └── synthetic_survey_data_bundling.csv   # Survey dataset (300 respondents × 35 cols)
```

---

## ⚙️ Setup & Run

### 1. Clone / download files
Place `chore_bundling_dashboard.py`, `requirements.txt`, and the CSV dataset in the same folder.

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place the dataset
Put `synthetic_survey_data_bundling.csv` in the path referenced in the script:
```
/mnt/user-data/uploads/synthetic_survey_data_bundling.csv
```
Or update the path on this line in `chore_bundling_dashboard.py`:
```python
df = pd.read_csv("/mnt/user-data/uploads/synthetic_survey_data_bundling.csv")
```

### 5. Launch the dashboard
```bash
streamlit run chore_bundling_dashboard.py
```
The app will open at `http://localhost:8501`

---

## 📊 Dataset Schema

| Column | Type | Description |
|---|---|---|
| `respondent_id` | str | Unique survey ID (R1000–R1299) |
| `area_dubai` | str | Dubai neighbourhood |
| `accommodation_type` | str | Dorm / Shared apartment / Private |
| `household_size` | int | Number of occupants |
| `monthly_budget_aed` | int | Declared monthly cleaning budget (AED) |
| `weekly_chore_hours` | float | Self-reported hours spent on chores per week |
| `academic_workload_1_5` | int | Likert 1–5 |
| `chore_interference_1_5` | int | Likert 1–5 — how much chores interfere with study |
| `cleanliness_satisfaction_1_5` | int | Likert 1–5 |
| `sel_room` … `sel_window` | int (0/1) | Binary chore selection flags (10 chores) |
| `num_chores_selected` | int | Count of selected chores |
| `q2_filler_item` … `q10_packaging_preference` | str | Qualitative survey questions |
| `exam_room` … `exam_dishwashing` | int (0/1) | Exam-week chore selection flags |
| `num_exam_chores_selected` | int | Count of exam-week chore selections |

---

## 🔬 Methodology

### Data Cleaning
- Duplicate respondent ID check and removal
- Missing value audit (confirmed 0 missing)
- Binary integrity validation on all `sel_*` columns
- Likert scale range enforcement (clip to 1–5)
- Budget outlier detection via 3×IQR rule
- `num_chores_selected` recomputed and cross-checked
- Categorical label whitespace normalisation

### Analytical Methods

| Method | Library | Purpose |
|---|---|---|
| Frequency & Co-occurrence Analysis | pandas, numpy | Demand ranking; pairwise selection matrix |
| Chi-Square Test of Independence | scipy.stats | Validate if accommodation type ↔ package preference is statistically significant |
| **Apriori Association Rules** | mlxtend | Market basket: which chores are requested together above chance. Metrics: Support, Confidence, Lift |
| **K-Means Clustering** (K=4) | scikit-learn | Segment respondents into 4 natural preference profiles → 4 package tiers |
| **RobustScaler** | scikit-learn | Scale features including budget column with outliers (median + IQR-based, not mean/std) |
| **Decision Tree** | scikit-learn | Interpretable baseline classifier for package preference prediction |
| **Random Forest** | scikit-learn | Ensemble classifier; feature importance for bundle driver identification |
| **Gradient Boosting** | scikit-learn | Boosted ensemble; best performance on small N tabular data |
| **Stratified K-Fold CV** (5-fold) | scikit-learn | Preserve class proportions across folds; avoid overfitting on small dataset |
| Precision & Recall | scikit-learn | TP/(TP+FP) and TP/(TP+FN) per package tier; confusion matrix on 80/20 split |

### Formula Reference
```
Support(A→B)    = P(A ∩ B)
Confidence(A→B) = P(B | A) = Support(A ∩ B) / Support(A)
Lift(A→B)       = Confidence(A→B) / Support(B)   [Lift > 1 = above-chance co-selection]

Precision       = TP / (TP + FP)
Recall          = TP / (TP + FN)
```

---

## 📦 Recommended Package Design (Output)

| Package | Core Chores | Price Band (AED) | Target Segment |
|---|---|---|---|
| **Starter Pack** | Room + Bathroom + Trash | 80–120 | Solo / dorm / light need |
| **Standard Clean** | Room + Bathroom + Kitchen + Dishwashing + Bedding | 180–250 | Shared apartment (3–4 pax) |
| **Full Care** | Standard Clean + Laundry + Ironing + Trash | 320–420 | Larger households, higher budget |
| **Exam Week Rescue** | Full Care + Grocery Run (surge pricing) | 280–350 | High academic workload, exam season |

### Key Bundle Rules (from Apriori)
- **Kitchen → Dishwashing**: Always bundle; highest confidence rule
- **Laundry + Ironing**: Never split; 59.7% of laundry users require them together
- **Window Cleaning**: Exclude from core bundles (15% demand); quarterly add-on only
- **Grocery Run**: Exam Rescue tier only; 50.3% of respondents have zero interest

---

## 📚 References

- Agrawal, R., & Srikant, R. (1994). *Fast algorithms for mining association rules.* VLDB Conference.
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
- Friedman, J.H. (2001). *Greedy function approximation: A gradient boosting machine.* Annals of Statistics, 29(5), 1189–1232.
- Lloyd, S.P. (1982). *Least squares quantization in PCM.* IEEE Transactions on Information Theory, 28(2), 129–137.
- Hubert, M., & Vandervieren, E. (2008). *An adjusted boxplot for skewed distributions.* Computational Statistics & Data Analysis, 52(12), 5186–5201. *(RobustScaler basis)*

---

*Dashboard built for Dubai student household services market validation — SP Jain MGB Digital Business Track*
