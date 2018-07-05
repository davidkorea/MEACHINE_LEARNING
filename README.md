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

# 3. Modeling
