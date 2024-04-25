#create tensorflow random forest model witih xgboost
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from xgboost import XGBClassifier

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = KerasClassifier(build_fn=lambda: XGBClassifier(), verbose=0)
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("XGBoost: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))   