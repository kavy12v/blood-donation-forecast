# 🩸 Blood Donation Forecast: Give Life, Predict Blood Donations
> **Slash Mark Machine Learning Internship — Task 4**  
> Predicting whether past blood donors will donate blood during future mobile donation drives using Machine Learning.

---

## 📌 1. Project Overview & Business Problem
Forecasting blood supply is a critical challenge in the healthcare ecosystem. Blood has a limited shelf life, yet shortages can put lives at immediate risk. As noted by the World Health Organization:
> *"Blood is the most precious gift that anyone can give to another person — the gift of life."*

Blood collection organizations (such as the Red Cross and Blood Transfusion Centers) frequently operate mobile donation vehicles. Predicting which past donors are likely to donate when a mobile blood drive visits helps managers:
- Target notifications and reminders cost-effectively.
- Optimize inventory and supply chain operations.
- Avoid both shortages and wasteful expiration of collected blood units.

---

## 📊 2. The Dataset (RFMTC Marketing Framework)
The dataset originates from the **UCI Machine Learning Repository** and records donor history from the Blood Transfusion Service Center in Hsin-Chu City, Taiwan. The center drives a mobile transfusion bus to universities every 3 months.

The data consists of **748 donor profiles** structured using the **RFMTC** (Recency, Frequency, Monetary, Time, Churn/Target) marketing model:

| Feature Name | Description | Units | Type |
| :--- | :--- | :--- | :--- |
| **Recency** | Months since the donor's last donation | Months | Numeric (Integer) |
| **Frequency** | Total number of donations made | Count | Numeric (Integer) |
| **Monetary** | Total volume of blood donated | c.c. (cubic centimeters) | Numeric (Integer) |
| **Time** | Months since the donor's very first donation | Months | Numeric (Integer) |
| **Target** | Whether the donor donated in March 2007 | 1 = Yes, 0 = No | Binary Categorical |

---

## 🔍 3. Key Observations & Exploratory Data Analysis (EDA)

1. **Target Class Imbalance**:
   - **Class 0 (Did not donate)**: 76.20% (570 donors)
   - **Class 1 (Donated blood)**: 23.80% (178 donors)
   - *Implication*: A naive classifier predicting "0" every time achieves 76.2% accuracy. Hence, **ROC AUC** (Area Under the Receiver Operating Characteristic Curve) is selected as the primary evaluation metric rather than simple accuracy.

2. **Extreme Variance Disparity**:
   - `Recency`: ~66.9
   - `Frequency`: ~33.8
   - `Time`: ~611.1
   - `Monetary (c.c. blood)`: **~2,114,363.7** (over 3 orders of magnitude larger!)
   - *Implication*: Linear and distance-based models would be heavily skewed towards `Monetary`. Applying **log-transformation** (`np.log(df['Monetary (c.c. blood)'])`) reduces the variance to **0.84**, stabilizing model gradients.

3. **Feature Redundancy**:
   - In this dataset, every blood donation corresponds to 250 c.c. of blood. Therefore, `Monetary` equals `Frequency × 250`. They are perfectly correlated ($r = 1.00$).

---

## ⚙️ 4. Machine Learning Workflow

```
1. Data Loading & Inspection (datasets/transfusion.data)
   └── Check null values, data types, and target distribution
2. Target Column Standardization
   └── Rename 'whether he/she donated blood in March 2007' -> 'target'
3. Stratified Train-Test Splitting
   └── 75% Training (561 samples), 25% Testing (187 samples) maintaining 76/24 ratio
4. Variance Analysis & Preprocessing
   └── Apply log-transformation to Monetary feature to handle extreme variance
5. Automated Model Selection (TPOT / Logistic Regression)
   └── TPOT search identified LogisticRegression as the optimal algorithm
6. Model Benchmarking & Multi-Algorithm Comparison
   └── Baseline LogReg vs Normalized LogReg vs Random Forest vs XGBoost
7. Evaluation & Metric Analysis
   └── ROC-AUC, Accuracy, Precision, Recall, F1-Score, Confusion Matrix
```

---

## 🏆 5. Model Evaluation & Benchmark Results

All models were evaluated on the held-out stratified test set (187 donors):

| Model | ROC AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normalized Logistic Regression** | **0.7891** | 77.54% | 0.6250 | 0.1136 | 0.1923 |
| **Baseline Logistic Regression** | 0.7851 | 77.01% | 0.5714 | 0.0909 | 0.1569 |
| **Random Forest Classifier** | 0.7678 | **80.75%** | **0.6429** | **0.4091** | **0.5000** |
| **XGBoost Classifier** | 0.7626 | 79.68% | 0.6071 | 0.3864 | 0.4722 |

### Key Takeaways:
- **Normalized Logistic Regression** achieved the highest **ROC-AUC score (0.7891)**, outperforming the baseline by 0.5% after log-normalizing the `Monetary` variable.
- For operational blood drives where ranking donors by probability is the goal (selecting top 20% to contact), **Logistic Regression with Log-Normalization** is the optimal choice.
- If raw balanced detection (higher recall of positive donors) is needed, **Random Forest** offers balanced F1-score performance (0.5000 with 40.91% recall).

---

## 📁 6. Project Directory Structure

```
Give-Life-Predict-Blood-Donations/
│
├── datasets/
│   └── transfusion.data              # UCI Blood Transfusion dataset (748 rows)
│
├── plots/                            # High-resolution generated charts
│   ├── 01_target_incidence.png        # Class imbalance distribution
│   ├── 02_feature_correlations.png    # Correlation heatmap
│   ├── 03_variance_log_transformation.png # Variance comparison before/after log transform
│   ├── 04_roc_auc_curves.png          # ROC curves comparison across models
│   ├── 05_confusion_matrix.png        # Best model confusion matrix
│   └── 06_feature_importance.png      # Logistic Regression coefficients
│
├── notebook.ipynb                    # Step-by-step interactive Jupyter Notebook (11 Tasks)
├── predict_blood_donations.py        # Automated end-to-end Python pipeline script
├── requirements.txt                  # Python dependencies
└── README.md                         # Detailed project documentation & report
```

---

## 🚀 7. How to Run This Project

### Method 1: Run the Complete Pipeline Script (Recommended)
Open your terminal / command prompt in this project folder and run:
```bash
# 1. Install dependencies (if not already installed)
pip install -r requirements.txt

# 2. Run the end-to-end ML pipeline
python predict_blood_donations.py
```
This will:
- Process the dataset.
- Train all models.
- Print the benchmark comparison table and classification report.
- Test interactive sample donor predictions.
- Generate and save all 6 charts in the `plots/` directory.

### Method 2: Run the Jupyter Notebook in VS Code
1. Open the project folder in **VS Code** (`File` > `Open Folder...`).
2. Open `notebook.ipynb`.
3. In the top-right corner of the notebook, select your Python kernel (Python 3.12).
4. Click **Run All** (or run each cell sequentially from Task 1 to Task 11).

---

## 🎓 8. Internship Submission Details
- **Internship Platform**: Slash Mark
- **Domain**: Machine Learning
- **Task ID**: Task 4 — Blood Donation Forecast
- **Technologies Used**: Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, XGBoost
