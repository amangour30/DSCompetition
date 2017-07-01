# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

dataset.dtypes
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# 1. Univariate Analysis

#continuous variables
dataset.describe()

# Getting unique count of each categories for categorical data
cv=dataset.dtypes.loc[dataset.dtypes=='object'].index
dataset[cv].apply(lambda x: len(x.unique()))

#print the count of each category
dataset['Country'].value_counts()


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)