# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# 1. Univariate Analysis: Investigating one variable

#continuous variables
dataset.describe()

# Getting unique count of each categories for categorical data
cv=dataset.dtypes.loc[dataset.dtypes=='object'].index
dataset[cv].apply(lambda x: len(x.unique()))

#print the count of each category
dataset['Country'].value_counts()

# 2. Multivariate Analysis: Comparing two or more variables
#print the cross tabulation
#both categorical
pd.crosstab(dataset['Country'], dataset['Purchased'], margins='true')

#both continuous then use the scatter plot
#compare continuous against categorical data
dataset.boxplot(column='Salary', by='Country')

#3. Check for missing values 
dataset.apply(lambda x: sum(x.isnull()))
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#4. Looking for outliers
# Use Scatter Plot and also look for values that doesn't make sense like -ve age
dataset.plot('Age','Salary',kind='Scatter')

#5. Variable Transformation
#5.1 Combing the variable (categorical) with very less percentages
# run a loop and combine all values in place
categories_to_combine=['France', 'Germany']
for cat in categories_to_combine:
    dataset['Country'].replace({cat:'CentralEUR', inplace=True})
#5.2 Creating a range for continuous variables 


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""