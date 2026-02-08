# Insurance Fraud Detection System using Machine Learning

## 1. Introduction

Insurance fraud is a major challenge for insurance companies due to the high volume of claims and the complexity involved in identifying fraudulent behavior. Manual fraud detection methods are inefficient, time-consuming, and prone to human error. As a result, there is a strong need for automated systems that can assist in identifying suspicious insurance claims accurately and efficiently.

This project presents a **machine learning–based insurance fraud detection system** that analyzes historical insurance claim data and predicts whether a claim is fraudulent or legitimate. The system evaluates and compares multiple supervised machine learning algorithms to identify the most effective model for fraud detection.

---

## 2. Problem Statement

Given a dataset of insurance claims containing customer, policy, and claim-related attributes, the goal is to build a system that can:

* Automatically classify insurance claims as **fraudulent** or **non-fraudulent**
* Reduce dependency on manual fraud investigation
* Improve detection accuracy using data-driven models
* Compare multiple ML algorithms to identify the best-performing model

---

## 3. System Overview

The system follows a **structured machine learning pipeline** consisting of data preprocessing, feature selection, dimensionality reduction, model training, and evaluation.

The complete workflow is implemented in Python using widely adopted machine learning libraries.

---

## 4. System Architecture and Flow

### Overall Flow

```
Insurance Claim Dataset
        ↓
Data Loading & Analysis
        ↓
Feature Selection
        ↓
Dimensionality Reduction (PCA)
        ↓
Train-Test Split
        ↓
Model Training (Multiple Algorithms)
        ↓
Model Evaluation
        ↓
Performance Comparison & Results
```

---

## 5. Methodology (How the system works)

### 5.1 Data Loading

The dataset (`main.csv`) is loaded using Pandas. Each row represents an insurance claim, and the target column `fraud_reported` indicates whether the claim is fraudulent.

* `0` → Non-fraud claim
* `1` → Fraudulent claim

---

### 5.2 Exploratory Data Analysis

Basic analysis is performed to understand:

* Total number of claims
* Distribution of fraudulent vs non-fraudulent claims
* Fraud percentage in the dataset

This helps in understanding the imbalance commonly present in fraud detection problems.

---

### 5.3 Feature Selection

To identify the most relevant attributes:

* **SelectKBest with ANOVA F-test** is applied
* Features are ranked based on their statistical relationship with the target variable

This step helps reduce irrelevant or weak features.

---

### 5.4 Dimensionality Reduction

**Principal Component Analysis (PCA)** is applied to:

* Reduce feature dimensionality
* Eliminate redundancy
* Improve computational efficiency

The dataset is reduced to **4 principal components**, preserving the most important variance in the data.

---

### 5.5 Train-Test Split

The dataset is split into:

* **80% training data**
* **20% testing data**

This ensures that model performance is evaluated on unseen data.

---

### 5.6 Model Training

The following supervised machine learning algorithms are trained:

* Decision Tree Classifier
* K-Nearest Neighbors (KNN)
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest Classifier
* XGBoost Classifier

Each model is trained using the same training dataset to ensure fair comparison.

---

### 5.7 Model Evaluation

Models are evaluated using:

* Accuracy Score
* Confusion Matrix

Confusion matrices provide deeper insight into:

* Correct fraud detections
* False positives
* False negatives

All confusion matrices are saved as images for analysis.

---

## 6. Results and Analysis

### 6.1 Dataset Summary

* Total claims: **1000**
* Non-fraud claims: **753**
* Fraud claims: **247**
* Fraud percentage: **32.8%**

---

### 6.2 Model Performance (Accuracy)

| Algorithm     | Accuracy  |
| ------------- | --------- |
| Decision Tree | 70.0%     |
| KNN           | 70.0%     |
| Random Forest | **71.5%** |
| XGBoost       | 63.5%     |

**Observation:**
Random Forest achieved the highest accuracy and showed better stability compared to other models.

---

### 6.3 Visual Performance Analysis

* Confusion matrices for each model are stored in:

  ```
  results/confusion_matrices/
  ```
* Overall model comparison chart is saved as:

  ```
  results/model_comparison.png
  ```

These visualizations help in understanding model behavior beyond numeric scores.

---

## 7. Project Structure

```
insurance-fraud-detection/
├── main_feature.py
├── Insurance_Fraud_Detection_Project_Report.docx
├── README.md
├── requirements.txt
├── PROJECT_STRUCTURE.md
├── .gitignore
├── data/
│   └── main.csv
└── results/
    ├── model_comparison.png
    └── confusion_matrices/
```

---

## 8. Technologies Used

* **Programming Language:** Python
* **Libraries:**

  * NumPy
  * Pandas
  * Matplotlib
  * Scikit-learn
  * XGBoost
  * Termcolor
* **IDE:** Visual Studio Code
* **Platform:** Windows 10 / 11

---

## 9. How to Run the Project

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main_feature.py
```

---

## 10. Outcomes

* Automated detection of fraudulent insurance claims
* Comparative analysis of multiple ML algorithms
* Identification of Random Forest as the most effective model
* Clear visualization of model performance
* Reduction in manual fraud analysis effort

---

## 11. Limitations

* Accuracy alone may not fully capture fraud detection performance
* Dataset is static and does not handle concept drift
* PCA reduces interpretability of original features

---

## 12. Future Enhancements

* Incorporate precision, recall, and F1-score
* Handle class imbalance using SMOTE or cost-sensitive learning
* Add explainable AI techniques (SHAP)
* Deploy the model as a real-time fraud detection API
* Extend system for live insurance claim processing

---

## 13. Conclusion

This project demonstrates the practical application of machine learning techniques in insurance fraud detection. By following a structured ML pipeline and comparing multiple classifiers, the system provides a reliable approach for identifying fraudulent claims. The project highlights the effectiveness of data-driven decision-making in reducing financial losses and improving operational efficiency in the insurance sector.

