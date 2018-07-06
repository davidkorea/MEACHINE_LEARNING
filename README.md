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

# 3. Modeling
