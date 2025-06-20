import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

df = pd.read_csv('SteelPlatesFaults.csv')
df.head()

df.info()
df.describe()
df.isnull().sum()

# If any one of the 7 error columns has a value of 1, then that product is faulty.
df['Fault'] = df.iloc[:, -7:].any(axis=1).astype(int)
df['Fault'].value_counts()

sns.countplot(x='Fault', data=df)
plt.title('Distribution of Faulty vs Non-Faulty Products')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.iloc[:, :-8].corr(), cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

X = df.drop(columns=df.columns[-8:])
y = df['Fault']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Most Important Features")
plt.show()


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='coolwarm', alpha=0.6)
plt.title("Product Distribution Using PCA")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.colorbar(label='Fault Status')
plt.show()