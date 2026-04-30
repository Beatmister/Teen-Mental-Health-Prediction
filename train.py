import numpy as np
import pandas as pd
import pre_processing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt

mental_health = pd.read_csv("Teen_Mental_Health_Dataset.csv")

processed_data = pre_processing.transform(mental_health)


X = processed_data.drop("depression_label", axis=1)
y = processed_data["depression_label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)