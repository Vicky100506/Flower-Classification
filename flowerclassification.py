# Iris Classification Project

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# 1. Load Dataset
# -------------------------------

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

print("\nFirst 5 rows of dataset:\n")
print(df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nStatistics:\n")
print(df.describe())

print("\nClass Distribution:\n")
print(df["species"].value_counts())

# -------------------------------
# 2. Exploratory Data Analysis
# -------------------------------

print("\nGenerating Pairplot...\n")
sns.pairplot(df, hue="species")
plt.show()

print("\nGenerating Correlation Heatmap...\n")
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation")
plt.show()

# -------------------------------
# 3. Prepare Data
# -------------------------------

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Train Models
# -------------------------------

print("\nTraining Logistic Regression...\n")

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_acc)

print("\nTraining Decision Tree...\n")

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy:", dt_acc)

# -------------------------------
# 5. Accuracy Comparison
# -------------------------------

print("\nModel Comparison\n")
print("Logistic Regression:", lr_acc)
print("Decision Tree:", dt_acc)

# -------------------------------
# 6. Confusion Matrix
# -------------------------------

print("\nConfusion Matrix (Logistic Regression)\n")

cm = confusion_matrix(y_test, lr_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# 7. CLI Prediction
# -------------------------------

def predict_species():
    print("\nEnter flower measurements")

    sepal_length = float(input("Sepal Length: "))
    sepal_width = float(input("Sepal Width: "))
    petal_length = float(input("Petal Length: "))
    petal_width = float(input("Petal Width: "))

    sample = [[sepal_length, sepal_width, petal_length, petal_width]]

    prediction = lr.predict(sample)

    print("\nPredicted Species:", iris.target_names[prediction][0])


# Run prediction loop
while True:
    choice = input("\nDo you want to predict a flower species? (y/n): ")

    if choice.lower() == "y":
        predict_species()
    else:
        print("Exiting program.")
        break