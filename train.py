import numpy as np
import pandas as pd
import pre_processing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

mental_health = pd.read_csv("Teen_Mental_Health_Dataset.csv")

processed_data = pre_processing(mental_health)

processed_data = pd.get_dummies(processed_data, columns=[""], drop_first=True)

X = processed_data.drop()
y = processed_data[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)