get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
data = pd.read_csv('https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/dataTrain_carListings.zip')
data.head()
data.shape
data.Price.describe()
data.plot(kind='scatter', y='Price', x='Year')
data.plot(kind='scatter', y='Price', x='Mileage')
data.columns
data_test = pd.read_csv('https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/dataTest_carListings.zip', index_col=0)
data_test.head()
data_test.shape
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
le = preprocessing.LabelEncoder()
feature_cols = ['Year', 'Mileage', 'State', 'Make', 'Model']
X = data[feature_cols]
y = data.Price
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in X.columns:
    if X[column_name].dtype == object:
        X[column_name] = le.fit_transform(X[column_name])
    else:
        pass
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,  test_size=0.3)

max_depth_range = range(1, 20)
accuracy_scores = []
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
for depth in max_depth_range:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=1)
    accuracy_scores.append(cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean())
import matplotlib.pyplot as plt
plt.plot(max_depth_range, accuracy_scores)
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
sorted(zip(accuracy_scores, max_depth_range))[::-1][0]

from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
lr = LogisticRegressionCV(cv = 5 )
lr.fit(X_train, y_train)
y_pred = lr.predict(y_test)
metrics.f1_score(y_pred, y_test), metrics.accuracy_score(y_pred, y_test)

import numpy as np
np.random.seed(42)
y_pred = pd.DataFrame(np.random.rand(data_test.shape[0]) * 75000 + 5000, index=data_test.index, columns=['Price'])
y_pred.to_csv('test_submission.csv', index_label='ID')
y_pred.head()

import gc
gc.collect()

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC 

models = {'lr': LinearRegression(),
          'dt': DecisionTreeRegressor(),
          'svm': SVC(kernel='linear')}
for model in models.keys():
    models[model].fit(X_train, y_train)
y_pred = pd.DataFrame(index=y_test.index, columns=models.keys())
for model in models.keys():
    y_pred[model] = models[model].predict(X_test)
from sklearn.metrics import mean_squared_error
for model in models.keys():
    print(model,np.sqrt(mean_squared_error(y_pred[model], y_test)))    
np.sqrt(mean_squared_error(y_pred.mean(axis=1), y_test))

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.distplot(data['Price'])
from scipy.stats import norm
import numpy as np
(mu, sigma) = norm.fit(data['Price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(data['Price'], plot=plt)
plt.show()

print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())

corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

cat = len(data.select_dtypes(include=['object']).columns)
num = len(data.select_dtypes(include=['int64','float64']).columns)
print('Total Features: ', cat, 'categorical', '+',
      num, 'numerical', '=', cat+num, 'features')

k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Price')['Price'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
most_corr
sns.jointplot(x=data['Year'], y=data['Price'], kind='reg')
