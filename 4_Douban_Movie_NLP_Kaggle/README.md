# Douban Movie NLP

# Issue 5 - word2vector

### 1. Bag of words (BoW)

**```from sklearn.feature_extraction.text import CountVectorizer```**
1. ```countvector = CountVectorizer()```
2. ```model_fit = countvector.fit(document)```
    - print(model_fit)
      ``` 
      CountVectorizer(analyzer='word', binary=False, decode_error='strict',
      dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
      lowercase=True, max_df=1.0, max_features=None, min_df=1,
      ngram_range=(1, 1), preprocessor=None, stop_words=None,
      strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
      tokenizer=None, vocabulary=None)
      ```
    - print(model_fit.vocabulary_)
      ``` 
      {'一条': 1, '天狗': 4, '日来': 5, '一切': 0, '星球': 6, '全宇宙': 3, '便是': 2}        
      ```


### 2. TF-IDF

**```from sklearn.feature_extraction.text import TfidfTransformer ```**



          
# Issue 4 - Pandas sample

- **```sample(n=None, frac=None, random_state=None)```**
    - n : int, optional, Number of items from axis to return. Cannot be used with frac. Default = 1 if frac = None.
    - frac : float, optional, Fraction of axis items to return. Cannot be used with n.
    - random_state : int, You can use random state for reproducibility.

```
>>> s = pd.Series(np.random.randn(50))
>>> s.head()
0   -0.038497
1    1.820773
2   -0.972766
3   -1.598270
4   -1.095526
dtype: float64

>>> df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))
>>> df.head()
          A         B         C         D
0  0.016443 -2.318952 -0.566372 -1.028078
1 -1.051921  0.438836  0.658280 -0.175797
2 -1.243569 -0.364626 -0.215065  0.057736
3  1.768216  0.404512 -0.385604 -1.457834
4  1.072446 -1.137172  0.314194 -0.046661
```
```
# 3 random elements from the Series:
>>> s.sample(n=3)
27   -0.994689
55   -1.049016
67   -0.224565
dtype: float64

# And a random 10% of the DataFrame with replacement:
>>> df.sample(frac=0.1, replace=True)
           A         B         C         D
35  1.981780  0.142106  1.817165 -0.290805
49 -1.336199 -0.448634 -0.789640  0.217116
40  0.823173 -0.078816  1.009536  1.015108
15  1.421154 -0.055301 -1.922594 -0.019696
6  -0.148339  0.832938  1.787600 -1.383767
```

# Issue 3 - Pandas dropna(subset=None)

- **```dropna(subset=None)```**
    - subset: Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include.
    - 删除某一行的时候，只有在subset指定的列中存在NaN时，执行删除此行
```
>>> df = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
...                    "toy": [np.nan, 'Batmobile', 'Bullwhip'],
...                    "born": [pd.NaT, pd.Timestamp("1940-04-25"), pd.NaT]})
>>> df
 name        toy       born
0    Alfred        NaN        NaT
1    Batman  Batmobile 1940-04-25
2  Catwoman   Bullwhip        NaT
```
```
>>> df.dropna(subset=['name', 'born'])
 name        toy       born
1    Batman  Batmobile 1940-04-25
```

# Issue 2 - re.compile(), re.sub()

1. **```re.compile(pattern[, flags])```**, compile 函数用于编译正则表达式，生成一个正则表达式（ Pattern ）对象，供 match(), search(), sub()函数使用

    - pattern : 一个字符串形式的正则表达式
    - flags 可选，表示匹配模式，比如忽略大小写，多行模式等，具体参数为：
        - re.I 忽略大小写
        - re.L 表示特殊字符集 \w, \W, \b, \B, \s, \S 依赖于当前环境
        - re.M 多行模式
        - re.S 即为' . '并且包括换行符在内的任意字符（' . '不包括换行符）
        - re.U 表示特殊字符集 \w, \W, \b, \B, \d, \D, \s, \S 依赖于 Unicode 字符属性数据库
        - re.X 为了增加可读性，忽略空格和' # '后面的注释
    - Ex: ```pattern = re.compile(r'([a-z]+) ([a-z]+)', re.I)   # re.I 表示忽略大小写```
    
2. **```re.sub(pattern, repl, string, count=0)```**, **替换 string中符合pattern规则的字符 为repl**， re.sub用于替换字符串中的匹配项
    - pattern : 正则中的模式字符串。
    - repl : 替换的字符串，也可为一个函数。
    - string : 要被查找替换的原始字符串。
    - count : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。
    - Ex: 
        ```python
        phone = "2004-959-559 # 这是一个电话号码"
        num = re.sub(r'\D', "", phone)  # 移除非数字的内容, 替换满足规则的字符为空
        print ("电话号码:", num)        
        [out]:
        电话号码:2004959559
        ```
    
3. Ex: 将raw_text中，符合filter_pattern规则的字符替换为''空，仅保留中文字符
```python
filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
chinese_only = filter_pattern.sub('', raw_text)
```

Reference: [Python3 正则表达式](http://www.runoob.com/python3/python3-reg-expressions.html)

# Issue 1 - Pandas apply() args parameter
**pandas: apply a function with arguments to a series**

You can pass any number of arguments to the function that apply is calling through either unnamed arguments, passed as a tuple to the args parameter, or through other keyword arguments internally captured as a dictionary by the kwds parameter.

For instance, let's build a function that returns True for values between 3 and 6, and False otherwise.

```python
s = pd.Series(np.random.randint(0,10, 10))
s

0    5
1    3
2    1
3    1
4    6
5    0
6    3
7    4
8    9
9    6
dtype: int64

s.apply(lambda x: x >= 3 and x <= 6)

0     True
1     True
2    False
3    False
4     True
5    False
6     True
7     True
8    False
9     True
dtype: bool
```
This anonymous function isn't very flexible. Let's created a normal function with two arguments to control the min and max values we want in our Series.
```python
def between(x, low, high):
    return x >= low and x =< high
```
We can replicate the output of the first function by passing unnamed arguments to args:
```python
s.apply(between, args=(3,6))
```
Or we can use the named arguments
```python
s.apply(between, low=3, high=6)
```
Or even a combination of both
```python
s.apply(between, args=(3,), high=6)

```
