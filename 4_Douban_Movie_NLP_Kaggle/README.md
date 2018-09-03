# Douban Movie NLP

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
