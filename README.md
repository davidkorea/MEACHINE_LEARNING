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


# 2. Feature Engineering

1.
data_df.dropna(inplace=True)

2.
data_df['length'].fillna(0, inplace=True)

3.
data_df.drop(['length','width'], axis=1, inplace=True)

4.
data_df['quality'].apply(lambda x:0 if x<6 else 1)


def level(x): # x is the each value of column 'quality'
  if x<6:
    labal=0
  else:
    label=1
  return label
data_df['quality'].apply(level)


data_df['level'] = pd.cut(x=data_df['quality'], bins=[-np.inf,3,5,np.inf], labels=['a','b','c','d'])



# 3. Modeling
