import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm
from sklearn.metrics import mean_absolute_error, confusion_matrix
import re
import numpy as np

# Load the data set
df = pd.read_csv("titanic.csv")

#print(df.describe())
#print(df.info())
#print(df.head(10))


## Missing data
total = df.isnull().sum().sort_values(ascending=False) # total number of values - else will count only NA values
percentageNull = ((df.isnull().sum()/ df.isnull().count()) * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percentageNull], axis=1, keys=['Total', 'MissingValues %'])
print(missing_data.head(10))

# Columns
print('Total number of columns are %d'%df.columns.values.size)
print(df.columns.values)

# Removing unnecessary columns
titanic_df = df.drop(['name'], axis=1) # Name is unnecessary
#print(titanic_df.head(10))

# Extracting the Deck out of the Cabin
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
titanic_df['cabin'] = titanic_df['cabin'].fillna('U0')
titanic_df['deck'] = titanic_df['cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
titanic_df['deck'] = titanic_df['deck'].map(deck)
titanic_df['deck'] = titanic_df['deck'].fillna(0)
titanic_df['deck'] = titanic_df['deck'].astype(int)
titanic_df = titanic_df.drop(['cabin'], axis=1)
#print(titanic_df.head(10))


# Replacing age with the mean
meanAge = titanic_df['age'].mean()
print('the mean age is %d'%meanAge)
ageCopy = titanic_df['age'].copy().replace(np.nan, meanAge, regex=True).apply(np.ceil).astype(int)
titanic_df['age'] = ageCopy

# Survived
titanic_df['survived'] = titanic_df['survived'].astype(int)

# Fare
titanic_df['fare'] = titanic_df['fare'].fillna(0).astype(int)


# categorical data
#print(titanic_df.info()) # sex, ticket, cabin, embarked, boat
#print(titanic_df.head(10))

#sex
genders = {'male':'M', 'female':'F'}
print(titanic_df['sex'].describe())
titanic_df['sex'] = titanic_df['sex'].map(genders)


# ticket - cannot convert to categories , too many unique tickets
titanic_df = titanic_df.drop(['ticket'], axis=1)


# Embarked
print(titanic_df['embarked'].describe())
titanic_df['embarked'] = titanic_df['embarked'].fillna('S') # Must fill dummies


#Boat
boat_null = titanic_df[pd.isna(titanic_df['boat'])]
print(boat_null['survived'].sum()) # Only 23 who did not get on a boat survived
titanic_df[['boat']] = np.where(titanic_df[['boat']].isnull(), 0, 1)

# Body - Too many missing values for body
print(titanic_df.head(5))


## Y = survived
## x = Features 
## No age - replace by average