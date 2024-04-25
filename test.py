#import svc
from sklearn import svm

#import cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#generate data  
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#convert dataset to dataframe
import pandas as pd
df_data = pd.DataFrame(iris.data)
df_target = pd.DataFrame(iris.target)
print(df_data)
print(df_target)

#generate SVC model
def test_SVC(X_train, X_test, y_train, y_test):
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)   
    return clf.score(X_test, y_test)

print(test_SVC(X_train, X_test, y_train, y_test))