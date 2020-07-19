import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm
from sklearn.metrics import mean_absolute_error, confusion_matrix

# Load the data set
df = pd.read_csv("titanic.csv")
print(df.head())