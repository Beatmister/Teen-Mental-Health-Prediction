import numpy as np
import pandas as pd
import pre_processing
from sklearn.model_selection import train_test_split

mental_health = pd.read_csv("Teen_Mental_Health_Dataset.csv")

processed_data = pre_processing(mental_health)

processed_data = pd.get_dummies(processed_data, columns=[""], drop_first=True)

X = processed_data.drop()
y = processed_data[""].values