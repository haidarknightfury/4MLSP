import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm
from sklearn.metrics import mean_absolute_error, confusion_matrix

# Load the data set
df = pd.read_csv("titanic.csv")

print(df.describe())
print(df.info())


print(df.head(10))

## Missing data
total = df.isnull().sum().sort_values(ascending=False)
print(total)

percentageNull = ((df.isnull().sum()/ df.isnull().count()) * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percentageNull], axis=1, keys=['Total', 'MissingValues %'])
print(missing_data.head(5))


## Drop unecessary variable such as name
## Y = survived
## x = Features 
## No age - replace by average