# MEACHINE_LEARNING 

1. EDA
2. Feature Engineering
3. Modeling

    3.1 Models & Params
    
        3.1.1 Supervised Learning
        
        3.1.2 Unsupervised Learning
    
    3.2 Regularization & Cross validation
    
    3.3 Ensemble learning

# 1. Exploration Data Analysis (EDA)

1. 
```python
pd.read_csv(path, index_col=None, usecols=None)
  - path = '../data/iris.csv'
  - index_col = 'Id'
  - usecols = ['length','width','species'] //list
```
2. 
```python
data_df.isnull().sum(axis=0)
```
3. 
```python
data_df['species'].value_counts()
```
4. 
```python
data_df['species'].value_counts().plot(kind='bar')
```
5. 
```python
sns.countplot(data=data_df,x='quality'), sns.countplot(x=data_df['quality'])
```
6. 
```python
FEAT_COLS = data_df.columns.tolist()[:-1]
```
7. 
```python
plt.scatter(x=data_df['SepalLengthCm'], y=data_df['SepalWidthCm'], c='r')
```
8. 
```python
corr = data_df[FEAT_COLS].corr()
plt.figure(figsize=(16,9))
sns.heatmap(data=corr, annot=True, cmap='coolwarm')
```
9.
```python
pd.scatter_matrix(data_df[FEAT_COLS],
                 diagonal='kde', # default=hist,
                 figsize=(16,9),
                 range_padding=0.1)
```

# 2. Feature Engineering

1.
```python
data_df.dropna(inplace=True)
```
2.
```python
data_df['length'].fillna(0, inplace=True)
```
3.
```python
data_df.drop(['length','width'], axis=1, inplace=True)
```
4.
```python
data_df['quality'].apply(lambda x:0 if x<6 else 1)
```

```python
def level(x): # x is the each value of column 'quality'
  if x<6:
    labal=0
  else:
    label=1
  return label
data_df['quality'].apply(level)
```

```python
data_df['level'] = pd.cut(data_df['HappScore'], bins=[-np.inf,3,5,np.inf], labels=['Low','Middle','High'])
```

```python
train_df['Sex'] = train_df['Sex'].map({'male':1,'female':0}) #replace original values by 0, 1
```
> For age, price data etc., get values distribution by data_df.describe(), then group it by Quartile(min,25%,50%,75%,max,mean)

5.
```python
all_df['platform_version'] = all_df['platform_version'].astype('str')
all_df['system'] = all_df['platform'].str.cat(all_df['platform_version'], sep='_')
```
6. 
- pd.concat, axis=0 => up+down, axis=1 => left+right
- do not need foreigner_key to link up together
```python
all_df = pd.concat([train_df,test_df],axis=0,ignore_index=True)
```
- pd.merge, left+right
- like sql link up together
```python
all_df = pd.merge(device_df, usage_df, on='user_id', how='inner')
```
7.
- ex:2010-06-28
```python
train_df['date_account_created'] = pd.to_datetime(train_df.date_account_created)
```
- ex:20090319043255
```python
tr_tfa_str = train_df['timestamp_first_active'].values.astype('str')
train_df['timestamp_first_active'] = pd.to_datetime(tr_tfa_str)
```
-
```python
df['tfa_year'] = np.array([x.year for x in df.timestamp_first_active])
```
-
```python
df['tfa_wd'] = np.array([x.isoweekday() for x in df.timestamp_first_active])
# return weekdays as 1,2,3,4,5,6,7 = mon ~ sun
```

8. One-hot encoding / Label encoding(each cate has one num)
```python
encoded_df = pd.get_dummies(df['dac_wd'], prefix='dac_wd')
df = pd.concat((df,encoded_df),axis=1)
```

9. Minmaxscaler
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_arr = scaler.fit_transform(data_df[FEAT_COLS])
scaled_df = pd.DataFrame(scaled_arr,columns=FEAT_COLS)
```

# 3. Modeling

```python
from sklearn.model_selection import train_test_split

X = data_df[FEAT_COLS].values
y = data_df['label'].values
# values不加也可以模型训练，算出score，但后面的X_test[idx, :]不可以进行slicing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 10)
```
> Issue: [What does the value of random_state mean? #2](https://github.com/davidkorea/MEACHINE_LEARNING/issues/2)

## 3.1 Supervised Learning
### 3.1.1 KNN
The nearest n neighbors, test sample's label = the majority labels of these n neighbors.

> **KNeighborsClassifier(n_neighbors=, p=)**
> - **n_neighbors** : int, optional (default = 5), Number of neighbors to use  
> - **p** : integer, optional (default = 2),
    Power parameter for the Minkowski metric. When p = 1, this is
    equivalent to using manhattan_distance (l1), and euclidean_distance
    (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

```python
from sklearn.neighbors import KNeighborsClassifier

k_list = [3,5,7]
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    print('k=', k , '-> Accuracy: ' ,acc)
```

### 3.1.2 Linear Regression
For Continuous value prediction

> **LinearRegression()**
> - **coef_** : coef_ is of shape (1, n_features), y = wx +b, w = coef_
> - **intercept_** : y = wx +b, b = intercept_

1. basic
```python
from sklearn.linear_model import LinearRegression

linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)
r2_score = linear_reg_model.score(X_test, y_test)
print(r2_score)
```
2. plot single feature scatter & regression line
```python
def plot_on_test(feat, coef, intercept, X_test, y_test):
    plt.scatter(X_test, y_test)
    plt.plot(X_test, coef*X_test+intercept, c='r')
    plt.title('Linear regression line of feature [ {} ] on test set'.format(feat))
    plt.show()
    
def plot_on_train(feat, coef, intercept, X_train, y_train):
    plt.scatter(X_train, y_train)
    plt.plot(X_train, coef*X_train+intercept, c='r')
    plt.title('Linear regression line of feature [ {} ] on train set'.format(feat))
    plt.show()  
    
# coef_ is of shape (1, n_features), that's why fit by each feature to get the only one coef to plot  
for feat in FEAT_COLS:
    X = house_df[feat].values.reshape(-1,1)
    y = house_df['price'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=10)
    lr_model = LinearRegression()
    lr_model.fit(X_train,y_train)
    r2 = lr_model.score(X_test,y_test)
    print(feat, ' -> r2 = ', r2)
    coef = lr_model.coef_
    intercept = lr_model.intercept_
    plot_on_test(feat, coef, intercept, X_test, y_test)
    plot_on_train(feat, coef, intercept, X_train, y_train)
    print('y = {}x + {}'.format(coef, intercept))

    print('-=*=-'*15)
    print()
```
![](https://raw.githubusercontent.com/davidkorea/MEACHINE_LEARNING/master/README/linearregressionplot.jpg)

[5_house_linear_regression_visualization](https://github.com/davidkorea/MEACHINE_LEARNING/blob/master/2_house_price/5_house_linear_regression_visualization.ipynb)

### 3.1.3 Logistic Regression
For classifying use. Comes from LinearREgression, and nonlinearize y = wx+b to ```y = (1+e^-z)^-1, z = wx+b, y∈（0, 1)```.

> **LogisticRegression(C=)**
> - **C** : float, default: 1.0. 
    Inverse of regularization strength; must be a positive float. 
    Like in support vector machines, smaller values specify stronger regularization.

```python
from sklearn.linear_model import LogisticRegression

linear_reg_model = LogisticRegression()
linear_reg_model.fit(X_train, y_train)
acc = linear_reg_model.score(X_test, y_test)
print(acc)
```
[Iris_classifier_logistic_svc](https://github.com/davidkorea/Iris_classifier_logistic_svc)

[6_iris_logistic_svc_complexity](https://github.com/davidkorea/MEACHINE_LEARNING/blob/master/1_iris/6_iris_logistic_svc_complexity.ipynb)

### 3.1.4 SVM

> **SVC(C=)**
> - **C** : float, optional (default=1.0).
    Penalty parameter C of the error term.误差项的惩罚参数.

```python
from sklearn.svm import SVC

model_dict = {
    'KNN':KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression':LogisticRegression(C=1e3),
    'SVC':SVC(C=1e3)
}

for model_name, model in model_dict.items():
    model.fit(X_train,y_train)
    acc = model.score(X_test, y_test)
    print(model_name, ' -> Accuracy = ',acc, '\n')
```

### 3.1.5 Neural Network - MLP
어려워ㅠㅠ deep Learning， TensorFlow， pytorch，BP，gradient，RNN, backpropagation，gradient descent 

- gradient descent 
对于二元，三元loss function，人可以解方程求得极值，但机器不会解方程，因此利用机器强大的计算能力进行逐步迭代。对于高维loss人也无能为力，只能依靠机器

### 3.1.6 Decision Tree

## 3.2 Unsupervised Learning
