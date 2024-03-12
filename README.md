# **Documentation Diabetes Prediction Project BNSP**
This project aims to predict whether someone has diabetes or not based on various health factors. I am use several machine learning algorithms and data preprocessing techniques to build an accurate model.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Approach](#2-approach)
3. [Data Collection](#3-data-collection)
4. [Data Preprocessing and Exploratory Data Analysis (EDA)](#4-data-preprocessing-and-exploratory-data-analysis-eda)
5. [Outlier Detection and Handling](#5-outlier-detection-and-handling)
6. [Data Splitting and Preparation](#6-data-splitting-and-preparation)
7. [Model Training](#7-model-training)
8. [Model Evaluation](#8-model-evaluation)
9. [Selecting the Best Model](#9-selecting-the-best-model)
10. [Conclusion](#10-conclusion)

## 1. Introduction

Diabetes is a disease that affects millions of people worldwide. By using machine learning technology, we aim to build a model that can predict the likelihood of someone having diabetes based on certain factors. This is expected to help in early identification and management of the disease. 

In this project, I am explore various machine learning algorithms to predict the likelihood of diabetes based on several health-related features. The goal is to build a predictive model that can assist in diagnosing diabetes. 

## 2. Approach
I am use a dataset containing patient health information, such as blood glucose, blood pressure, and BMI, to train multiple machine learning models. I am perform data preprocessing, splitting the dataset into training and testing data, and evaluating model performance using metrics such as accuracy, precision, recall, and F1-score.

## 3. Data Collection

We obtained the dataset from [this source](https://raw.githubusercontent.com/arubhasy/dataset/main/diabetes.csv) or can download the dataset on repository.
```python
import pandas as pd

# Mengimpor dataset diabetes
df = pd.read_csv('https://raw.githubusercontent.com/arubhasy/dataset/main/diabetes.csv')
```

## 4. Data Preprocessing and Exploratory Data Analysis (EDA)

We performed data preprocessing tasks such as handling missing values, checking for duplicates, and exploring the distribution of features. This step also includes visualizing descriptive statistics and histograms for each attribute.

### Data Processing
```python
# Menampilkan informasi dataset
df.info()

# Mengecek missing values
df.isna().sum()

# Mengisi nilai NaN dengan 0
df.fillna(0, inplace=True)

# Mengecek data duplikat
df.duplicated().sum()
```

### Exploratory Data Analysis (EDA)
```python
# Menampilkan statistik deskriptif
descriptive_stats = df.describe()
print(descriptive_stats)

# Membuat visualisasi rata-rata
plt.figure(figsize=(10, 3))
plt.bar(descriptive_stats.columns, descriptive_stats.loc['mean'], color='skyblue')
plt.title("Mean Values for Each Attribute")
plt.xlabel("Attribute")
plt.ylabel("Mean Value")
plt.xticks(rotation=45)
plt.show()

```

## 5. Outlier Detection and Handling

We detected and handled outliers in the dataset using methods such as Z-Score and IQR to ensure data quality before training the model.
```python
# Mendeteksi skewness dalam setiap fitur
skewness = df.skew()

# Menampilkan skewness untuk setiap atribut
print(skewness)

# Menampilkan histogram untuk setiap fitur
plt.figure(figsize=(12, 12))
for i, col in enumerate(df.columns):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[col], bins=30, color='skyblue', kde=True)
    skewness_label = 'Positif Skewness' if skewness[col] > 0.5 else ('Negatif Skewness' if skewness[col] < -0.5 else 'Mendekati Normal')
    plt.title(f'Histogram of {col}\n({skewness_label})')
plt.tight_layout()
plt.show()

# Menangani outlier dengan Z-Score atau IQR Method
# (Kode untuk penanganan outlier tidak disertakan di sini untuk ringkasnya)

```

## 6. Data Splitting and Preparation

We split the data into training and testing sets to train and evaluate the machine learning models.
```python
from sklearn.model_selection import train_test_split

# Memisahkan fitur dan target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

## 7. Model Training

We trained multiple machine learning models, including K-Nearest Neighbors (KNN), Decision Tree, Random Forest, and Adaboost, using the training data.
```python
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Inisialisasi model
knn_model = KNeighborsClassifier()
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()
ab_model = AdaBoostClassifier()

# Melatih model
knn_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
ab_model.fit(X_train, y_train)
```

## 8. Model Evaluation

We evaluated the trained models using metrics such as accuracy, precision, recall, and F1-score on the testing data.
```python
from sklearn.metrics import classification_report, confusion_matrix

# Evaluasi model KNN
y_pred_knn = knn_model.predict(X_test)
print("K-Nearest Neighbors:")
print(classification_report(y_test, y_pred_knn))

# Evaluasi model Decision Tree
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree:")
print(classification_report(y_test, y_pred_dt))

# Evaluasi model Random Forest
y_pred_rf = rf_model.predict(X_test)
print("Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Evaluasi model Adaboost
y_pred_ab = ab_model.predict(X_test)
print("Adaboost:")
print(classification_report(y_test, y_pred_ab))
```

## 9. Selecting the Best Model

Based on the evaluation results, we selected the best-performing model to predict new instances of diabetes.
```python
best_model = rf_model  # Misalnya, menggunakan Random Forest sebagai model terbaik
```

## 10. Conclusion

In conclusion, the Random Forest model demonstrated the best performance in predicting diabetes cases. Therefore, we chose the Random Forest model for predicting diabetes cases in new data instances.

---

This documentation provides a comprehensive overview of each step performed in the diabetes prediction project using machine learning. You can further customize it by adding additional details or explanations as needed.
