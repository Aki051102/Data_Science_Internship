import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")
gender_submission_df = pd.read_csv("dataset/gender_submission.csv")

train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())
train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())
train_df.drop(columns=["Cabin", "Name", "Ticket"], inplace=True)
test_df.drop(columns=["Cabin", "Name", "Ticket"], inplace=True)

train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})
test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})
train_df["Embarked"] = train_df["Embarked"].map({"C": 0, "Q": 1, "S": 2})
test_df["Embarked"] = test_df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

sns.countplot(x="Survived", data=train_df)
plt.show()

sns.countplot(x="Survived", hue="Sex", data=train_df)
plt.show()

sns.countplot(x="Survived", hue="Pclass", data=train_df)
plt.show()

sns.histplot(train_df["Age"], bins=20, kde=True)
plt.show()

plt.figure(figsize=(10, 6))
numeric_cols = train_df.select_dtypes(include=[np.number])
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.show()

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X_train = train_df[features]
y_train = train_df["Survived"]
X_test = test_df[features]

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_split, y_train_split)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

test_predictions = model.predict(X_test)
submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": test_predictions})
submission.to_csv("submission.csv", index=False)
print("Submission file created!")
