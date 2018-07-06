# MEACHINE_LEARNING

1. EDA
2. Feature Engineering
3. Modeling

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
5.
```python
all_df['platform_version'] = all_df['platform_version'].astype('str')
all_df['system'] = all_df['platform'].str.cat(all_df['platform_version'], sep='_')
```
6. 
- pd.concat, axis=0 => up+down, axis=1 => left+right
```python
all_df = pd.concat([train_df,test_df],axis=0,ignore_index=True)
```
- pd.merge, left+right
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

8. One-hot encoding
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
> - **coef_** : y = wx +b, w = coef_
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
[5_house_linear_regression_visualization.ipynb](https://github.com/davidkorea/MEACHINE_LEARNING/blob/master/2_house_price/5_house_linear_regression_visualization.ipynb)

### 3.1.3 Logistic Regression


### 3.1.4 SVM


### 3.1.5 Neural Network - MLP


### 3.1.6 Decision Tree


## 3.2 Unsupervised Learning
