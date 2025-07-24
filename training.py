import pandas as pd
df = pd.read_csv("customer_churn_dataset-training-master.csv")

# Perulangan untuk melihat kolom yang memiliki nilai null, jumlah baris null di kolom tersebut, persentase null di kolom tersebut, dan nilai unik

df_count = len(df)
col_null = [col_name for col_name in df.columns if df[col_name].isnull().any()]

for column in col_null:
    nulls_count = df[column].isnull().sum()

    print("")
    print(f"-- Kolom {column} --")
    print("")
    print(f"Jumlah Null: {nulls_count}")
    print(f"Persentase Null: {nulls_count / df_count * 100:.2f}%")
    print(f"Nilai unik: {df[column].dropna().unique()}")
    print("")
    print("=====================================================")

# Melihat baris yang memiliki nilai null
df[df.isnull().any(axis=1)]

# Menghapus nilai null
df = df.dropna()

# Memastikan nilai null sudah terhapus(1)
df[df.isnull().any(axis=1)]

# Memastikan nilai null sudah terhapus(2)
print([col_name for col_name in df.columns if df[col_name].isnull().any()])

# Memastikan nilai null sudah terhapus(3)
df.shape[0]

# Melihat apakah ada data yang duplikat
grouped_df = df.groupby(df.columns.tolist()).size().reset_index(name='count')
filtered_df = grouped_df[grouped_df['count'] > 1]

filtered_df.head()

# Membagi dua list, data kategorikal dan data numerikal
categorical_col = df.select_dtypes(include=['object']).columns.tolist()
numerical_col = df.select_dtypes(include=['float']).columns.tolist()

print(f"Fitur Kategorikal: {categorical_col}")
print(f"Fitur Numerik: {numerical_col}")

# Outlier Data Kategorikal
print("-- Nilai Unik --")
for column in categorical_col:
    print("")
    print(f"Kolom {column}: {df[column].drop_duplicates().tolist()}")

categorical_data = df[categorical_col]

# Bar Plot
import matplotlib.pyplot as plt
import numpy as np

for column in categorical_data:
    value_counts = categorical_data[column].value_counts(normalize=True)

    plt.figure(figsize=(4, 3))
    plt.bar(value_counts.index, value_counts.values)
    plt.title(f"Distribusi fitur {column}")
    plt.xlabel("Kategori")
    plt.ylabel("Jumlah Data")

# Outlier Data Numerik

print(f"-- Outlier --")
for column in numerical_col:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outliers_count = outliers.shape[0]

    print(f"")
    print(f"Kolom: {column}")
    print(f"Jumlah: {outliers_count}")
    print(f"Persentase: {outliers_count / df.shape[0] * 100}%")

categorical_data.head(5)

# Ordinal
ordinal_data = categorical_data[['Subscription Type', 'Contract Length']]
ordinal_data.head(5)

for column in ordinal_data:
    print(f"Kolom {column}: {df[column].drop_duplicates().tolist()}")

# Mapping manual
ordinal_data['Subscription Type Index'] = ordinal_data['Subscription Type'].map({'Basic': 1, 'Standard': 2, 'Premium': 3})
ordinal_data['Contract Length Index'] = ordinal_data['Contract Length'].map({'Monthly': 1, 'Quarterly': 2, 'Annual': 3})

# Drop kolom lama
ordinal_data = ordinal_data.drop(columns=["Subscription Type", "Contract Length"])

# Buat id untuk menyatukan kembali data
ordinal_data['id'] = range(len(ordinal_data))

ordinal_data.head(5)

# Nominal
nominal_data = categorical_data[['Gender']]
nominal_data.head(5)

# Encoding
encoded_nominal = pd.get_dummies(nominal_data, columns=['Gender'], drop_first=False)

# Buat id untuk menyatukan kembali data
encoded_nominal['id'] = range(len(encoded_nominal))

# Mengubah tipe data hasil encoding dari boolean ke integer
encoded_nominal[['Gender_Female', 'Gender_Male']] = encoded_nominal[['Gender_Female', 'Gender_Male']].astype(int)

encoded_nominal.head(5)

df['Gender'].value_counts()

numerical_data = df[numerical_col]

numerical_data.head(5)

df['Last Interaction'].nunique()

# Melihat distribusi data

for column in numerical_data:
    data = numerical_data[column]

    skewness = data.skew()
    print(f"Skewness kolom {column}: {skewness:.4f}")

sc = numerical_data['Support Calls']

sc_skewness = sc.skew()
sturges_bin_sc = int(1 + np.log2(len(sc)))

plt.figure(figsize=(8, 6))
plt.hist(sc, bins=sturges_bin_sc)
plt.title(f"Distribusi fitur Support Calls\nSkewness: {sc_skewness:.4f}")
plt.xlabel("Value")
plt.ylabel("Jumlah Value")
plt.show()

# Transformasi Kolom Support Calls dengan Log Transform

numerical_data['Support Calls_Log'] = np.log1p(numerical_data['Support Calls'])

scl = numerical_data['Support Calls_Log']

scl_skewness = scl.skew()
sturges_bin_scl = int(1 + np.log2(len(scl)))

plt.figure(figsize=(8, 6))
plt.hist(scl, bins=sturges_bin_scl)
plt.title(f"Distribusi fitur Support Calls_Log\nSkewness: {scl_skewness:.4f}")
plt.xlabel("Value")
plt.ylabel("Jumlah Value")
plt.show()

numerical_data = numerical_data.drop("Support Calls", axis=1)

numerical_data.head(5)

# Buat id untuk menyatukan kembali data
numerical_data['id'] = range(len(numerical_data))

# Menyatukan data kategorik ordinal dengan data numerik

data_normalized = numerical_data.merge(ordinal_data, on="id")
data_normalized = data_normalized.drop("id", axis=1)

data_normalized.head(5)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
columns = data_normalized.columns

data_normalized = scaler.fit_transform(data_normalized)

data_normalized = pd.DataFrame(data_normalized, columns=columns)

data_normalized.head(5)

# Buat id untuk menyatukan kembali data
data_normalized['id'] = range(len(data_normalized))

# Menyatukan lagi semua data

final_data = data_normalized.merge(encoded_nominal, on="id")

final_data = final_data.drop("id", axis=1)

final_data.head(5)

# Drop CustomerID, Gender_Female, dan Gender_Male karena merupakan fitur yang tidak relevan

final_data = final_data.drop(columns=['CustomerID', 'Gender_Female', 'Gender_Male'])

final_data.head(5)

target_col = 'Churn'
feature_cols = final_data.columns.drop(target_col).tolist()

from scipy.stats import pointbiserialr
import seaborn as sns

corr_matrix = pd.DataFrame(index=feature_cols, columns=['Correlation'])
pval_matrix = pd.DataFrame(index=feature_cols, columns=['P-Value'])

for col in feature_cols:
    corr, pval = pointbiserialr(final_data[col], final_data[target_col])
    corr_matrix.loc[col, 'Correlation'] = corr
    pval_matrix.loc[col, 'P-Value'] = pval

corr_matrix = corr_matrix.astype(float)
pval_matrix = pval_matrix.astype(float)

plt.figure(figsize=(6, 4))
sns.heatmap(pval_matrix, annot=True, cmap='YlGnBu', center=0)
plt.title("P-Value")
plt.show()
print("")
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Point-Biserial Correlation")
plt.show()

for column in final_data:
    data = final_data[column]

    skewness = data.skew()
    print(f"Skewness kolom {column}: {skewness:.4f}")

features = final_data[feature_cols]

from scipy.stats import pearsonr

pearson_corr = features.corr(method='pearson')

plt.figure(figsize=(8, 6))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson Correlation Heatmap")
plt.show()

spearman_corr = features.corr(method='spearman')

plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Spearman Correlation Heatmap")
plt.show()

# Drop kolom Subscription Type Index, Usage Frequency, Tenure, dan Last Interaction karena korelasi dengan Churn kecil
# Tersisa kolom Age, Support Calls, Payment Delay, Total Spend, Contract Length Index, dan Churn

selected_data = final_data[['Age', 'Support Calls_Log', 'Payment Delay', 'Total Spend', 'Contract Length Index', 'Churn']]
selected_data.head(5)

# Melihat korelasi antar fitur yang terseleksi

from scipy.stats import pearsonr

pearson_corr_selected = selected_data.corr(method='pearson')
spearman_corr_selected = selected_data.corr(method='spearman')

plt.figure(figsize=(8, 6))
sns.heatmap(pearson_corr_selected, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson Correlation Heatmap")
plt.show()
print("")
plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr_selected, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Spearman Correlation Heatmap")
plt.show()

# Menmbagi data jadi feature dan target

selected_features = selected_data.drop(columns=['Churn'])

selected_features.head(5)

selected_target = selected_data['Churn']

selected_target.head(5)

selected_features.dtypes

selected_features.rename(columns={
    "Support Calls_Log": "Support Calls",
    "Contract Length Index": "Contract Length"
}, inplace=True)

selected_features.head(5)

# Memastikan ada tidaknya multikolinearitas dg Variance Inflation Factor (VIF)

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_const = add_constant(selected_features)

vif_df = pd.DataFrame()
vif_df["Feature"] = X_const.columns
vif_df["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

print(vif_df)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

X = selected_features
y = selected_target

print("Distribusi Target:\n")
print(y.value_counts(normalize=True))

baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'Accuracy': 'accuracy',
    'Precision': 'precision_macro',
    'Recall': 'recall_macro'
}

baseline_cv_result = cross_validate(baseline_model, X, y, cv=cv, scoring=scoring)

print("Hasil K-Fold Evaluation:\n")
for metric in scoring.keys():
    scores = baseline_cv_result[f'test_{metric}']
    print(f"{metric}: mean = {np.mean(scores):.4f}, std = {np.std(scores):.4f}")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

baseline_model.fit(x_train, y_train)
y_pred = baseline_model.predict(x_test)

print("Evaluasi di Test Set:\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

train_acc = accuracy_score(y_train, baseline_model.predict(x_train))
test_acc = accuracy_score(y_test, y_pred)
print(f"\nPerbandingan Akurasi:\nTrain: {train_acc:.4f}, Test: {test_acc:.4f}")

importances = baseline_model.feature_importances_
feat_names = X.columns if hasattr(X, 'columns') else [f'F{i}' for i in range(X.shape[1])]
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.title("Feature Importances (Baseline Model)")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feat_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(x_train, y_train)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

y_pred = final_model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:\n", accuracy_score(y_test, y_pred))
print("Classification Report:\n\n", classification_report(y_test, y_pred))
print(f"Confusion Matriks:\n\n{cm}")

import joblib

joblib.dump(final_model, "model.pkl")
print("Model saved!")

joblib.dump(scaler, "scaler.pkl")
print("Scaler saved!")