import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

 
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

 
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

print("Train data missing values:", train_data.isnull().sum())
print("Test data missing values:", test_data.isnull().sum())

train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())
test_data["Age"] = test_data["Age"].fillna(test_data["Age"].mean())

train_data["Embarked"] = train_data["Embarked"].fillna("S")
test_data["Embarked"] = test_data["Embarked"].fillna("S")

sns.countplot(x="Sex", data=train_data)
plt.show()

sns.countplot(x="Pclass", hue="Survived", data=train_data)
plt.show()

sns.distplot(train_data["Age"])
plt.show()

sns.boxplot(
    x="Pclass",
    y="Age",
    showmeans=True,
    data=train_data,
)
plt.show()

sns.scatterplot(x="Age", y="Fare", data=train_data)
plt.show()

sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train_data)
plt.show()

print("Survival rate by gender:")
print(train_data["Survived"].groupby("Sex").mean())

print("Survival rate by Pclass:")
print(train_data["Survived"].groupby("Pclass").mean())

print("Correlation matrix:")
print(train_data.corr())
