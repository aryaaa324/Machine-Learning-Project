# Heart Disease Prediction Using Machine Learning

## 📌 Introduction
Heart disease is one of the leading causes of mortality globally. Early detection is crucial to preventing life-threatening cardiac events. This project analyzes patient medical attributes such as **age, blood pressure, cholesterol, heart rate, chest pain type**, etc., to predict the likelihood of heart disease.

The primary goal is to **build a predictive machine learning model** with **high recall for class 1 (patients with heart disease)** to ensure we **do not miss diagnosing a patient at risk**.

---

## 🎯 Problem Statement
Develop a machine learning model that predicts whether a patient is at risk of heart disease based on medical measurements, focusing on **maximizing recall** to identify as many heart-disease patients as possible.

---

## 🎯 Objectives
* **Explore the Dataset** to understand feature distributions and behavior
* Perform **Extensive EDA** — univariate and bivariate analysis with respect to the target
* Apply **Data Preprocessing**, including:
  - Removal of irrelevant attributes
  - Handling missing values and outliers
  - Encoding categorical features
  - Scaling & normalizing skewed attributes
* **Model Training** using:
  - Decision Tree
  - Random Forest
  - KNN
  - SVM
* **Optimize hyperparameters** for improved performance
* **Evaluate models** with Precision, Recall, and F1-Score (Recall prioritized)

---

## 🧭 Table of Contents  
**(As provided — unchanged and preserved exactly)**

* [Step 1 | Import Libraries](#import)
* [Step 2 | Read Dataset](#read)
* [Step 3 | Dataset Overview](#overview)
    - [Step 3.1 | Dataset Basic Information](#basic)
    - [Step 3.2 | Summary Statistics for Numerical Variables](#num_statistics)
    - [Step 3.3 | Summary Statistics for Categorical Variables](#cat_statistics)
* [Step 4 | EDA](#eda)
    - [Step 4.1 | Univariate Analysis](#univariate)
        - [Step 4.1.1 | Numerical Variables Univariate Analysis](#num_uni)
        - [Step 4.1.2 | Categorical Variables Univariate Analysis](#cat_uni)
    - [Step 4.2 | Bivariate Analysis](#bivariate)
        - [Step 4.2.1 | Numerical Features vs Target](#num_target)
        - [Step 4.2.2 | Categorical Features vs Target](#cat_target)
* [Step 5 | Data Preprocessing](#preprocessing)
    - [Step 5.1 | Irrelevant Features Removal](#feature_removal)
    - [Step 5.2 | Missing Value Treatment](#missing)
    - [Step 5.3 | Outlier Treatment](#outlier)
    - [Step 5.4 | Categorical Features Encoding](#encoding)
    - [Step 5.5 | Feature Scaling](#scaling)
    - [Step 5.6 | Transforming Skewed Features](#transform)
* [Step 6 | Decision Tree Model Building](#dt)
    - [Step 6.1 | DT Base Model Definition](#dt_base)
    - [Step 6.2 | DT Hyperparameter Tuning](#dt_hp)
    - [Step 6.3 | DT Model Evaluation](#dt_eval)
* [Step 7 | Random Forest Model Building](#rf)
    - [Step 7.1 | RF Base Model Definition](#rf_base)
    - [Step 7.2 | RF Hyperparameter Tuning](#rf_hp)
    - [Step 7.3 | RF Model Evaluation](#rf_eval)
* [Step 8 | KNN Model Building](#knn)
    - [Step 8.1 | KNN Base Model Definition](#knn_base)
    - [Step 8.2 | KNN Hyperparameter Tuning](#knn_hp)
    - [Step 8.3 | KNN Model Evaluation](#knn_eval)
* [Step 9 | SVM Model Building](#svm)
    - [Step 9.1 | SVM Base Model Definition](#svm_base)
    - [Step 9.2 | SVM Hyperparameter Tuning](#svm_hp)
    - [Step 9.3 | SVM Model Evaluation](#svm_eval)
* [Step 10 | Conclusion](#conclusion)

---

## 🔍 Dataset Overview
The dataset consists of medical records used to determine heart disease likelihood.

| Feature | Description |
|--------|-------------|
| Age | Age of the patient |
| Sex | 1 = Male, 0 = Female |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol level |
| Fasting Blood Sugar | Indicator >120mg/dl |
| Chest Pain Type | Medical category for chest pain |
| Max Heart Rate | Achieved during exercise |
| Other Cardiac Indicators | ECG results, exercise-induced angina, etc. |
| **Target** | 1 = Heart Disease, 0 = No Disease |

---

## 🧠 Model Building & Performance Summary

| Model | Needs Scaling | Key Strength | Notes |
|------|-------------|-------------|------|
| Decision Tree | ❌ | Easy to interpret | Prone to overfitting |
| Random Forest | ❌ | **High recall & robustness** | Best overall model |
| KNN | ✅ | Simple, distance-based | Highly scale-sensitive |
| SVM | ✅ | Strong boundary classification | Recall depends on kernel selection |

**Metric Focus:**  
⚠️ Recall (Sensitivity) for class 1 (heart disease present) was prioritized.

---

## ✅ Conclusion
- **Random Forest** performed best with **high recall**, making it suitable for medical early-warning systems.
- Proper **data preprocessing** significantly improved model reliability.
- The model can assist healthcare professionals by supporting early risk screening.

### 🔮 Future Scope:
- Larger real-world clinical datasets
- Deep learning & ensemble stacking
- Web-app deployment for clinical usage

---

## 📝 License
This project is licensed under the **MIT License**.

