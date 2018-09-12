
## Matrices and Operations

## Objectives

* Describe a matrix as a combination of scalar values in multiple dimensions.
* Define a matrix in python and using indexing to address/modify its elements.
* Apply basic matrix-matrix and vector-matrix arithmetic operations.
* Calculate dot products, matrix transpose for solving systems of equatrions.

## Introduction

A matrix is a rectangular array of numbers written between square brackets. As compared to vectors, a matrix is a multi-dimensional array of scalars that can possibly have multiple rows as well as columns. It usually denoted by an **m x n** notation where n is the number of rows and M is number of columns. A matrix is usually written down as:

$$
\left(\begin{array}{cc} 
m1 & m2 & m3 \\
m4 & m5 & m6 \\
m7 & m8 & m9
\end{array}\right)
$$

Here m1,..,m9 are scalar values. 

In machine learning, a vector is known to be a special case of a matrix. A vector is a matrix that has only 1 column so you have an N x 1 matrix. N is the number of rows, and 1 here is the number of columns, so, so matrix with just one column is what we call a vector. 

We counter matrices in machine learning during model training where data is comprised of many rows and columns and often represented using the capital letter “X”.

Often the dimensions of the matrix are denoted as m and n for the number of rows and the number of columns.

$$
Matrix (m x n ) = 
\left(\begin{array}{cc} 
1,1 & 1,2 & 1,3 \\
2,1 & 2,2 & 2,3 \\
3,1 & 3,2 & 3,3 
\end{array}\right)
$$

Now that we know what a matrix is, let’s look at defining one in Python.



### Defining a Matrix in Python

As opposed to one-dimensional arrays used by vectors, We can represent a matrix in Python using a multi-dimensional NumPy array. A NumPy array can be constructed given a list of lists. For example, below is a 3 row, 3 column matrix being created from a list of three lists. 


```python
import numpy as np
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(X)
```

    [[1 2 3]
     [4 5 6]
     [7 8 9]]


Note: Take special care with brackets during definition as opening and closing of the square brackets signifies a new row. 

### Matrix Indexing 

In 2d arrays like the one we created above, we use a row, column notation. We use a `:` to indicate all rows or all columns. Also keep in mind that the indexing in both vectors and matrices start at `0` and finishes at `n-1` and `m-1`.


```python
print (X[0, 0]) # element at first row and first column

print (X[-1, -1]) # elemenet last row and last column 

print (X[0, :]) # first row and all columns

print (X[:, 0]) # all rows and first column 

print (X[:]) # all rows and all columns
```

    1
    9
    [1 2 3]
    [1 4 7]
    [[1 2 3]
     [4 5 6]
     [7 8 9]]


We can also use indexing to address and assign new values to elements of a matrix as shown below:


```python
X[:, 0] = [11, 12, 13] # set column 0
X[2, 2] = 15          # set a single element in third row and thirs column
print (X)

X[2] = 16  # sets everything in row 3 to 16!
print (X)

X[:,2] = 17  # sets everything in column 3 to 17!
print (X)
```

    [[11  2  3]
     [12  5  6]
     [13  8 15]]
    [[11  2  3]
     [12  5  6]
     [16 16 16]]
    [[11  2 17]
     [12  5 17]
     [16 16 17]]


### Basic Matrix Arithmetic 

We shall now look at simple matrix-matrix arithmetic, where all operations are performed element-wise between two matrices of equal size to result in a new matrix with the same size - very similar to vector arithmetic that saw earlier. 

#### Addition
Two matrices with the same dimensions can be added together to create a new third matrix as:

```
         a11 + b11, a12 + b12
A + B = (a21 + b21, a22 + b22)
         a31 + b31, a32 + b32
```

here `a(nm)` and `b(nm)` represent row and column locations. We can perform this operation in Numpy as:


```python
# create and add two matrices
M1 = np.array([[1, 2, 3], [4, 5, 6]])
M2 = np.array([[7, 8, 9], [10, 11, 12]])

M = M1 + M2

print(M1, '\n+\n', M2, '\n=\n',M)
```

    [[1 2 3]
     [4 5 6]] 
    +
     [[ 7  8  9]
     [10 11 12]] 
    =
     [[ 8 10 12]
     [14 16 18]]


#### Subtraction 

A matrix can be subtracted from another matrix having the same dimensions as:
```
         a11 - b11, a12 - b12
A - B = (a21 - b21, a22 - b22)
         a31 - b31, a32 - b32

```


```python
# create two matrices and subtract one from the other
M1 = np.array([[1, 2, 3], [4, 5, 6]])
M2 = np.array([[7, 8, 9], [10, 11, 12]])

M = M2 - M1

print(M2, '\n-\n', M1, '\n=\n',M)
```

    [[ 7  8  9]
     [10 11 12]] 
    -
     [[1 2 3]
     [4 5 6]] 
    =
     [[6 6 6]
     [6 6 6]]


#### Multiplication
Two matrices with the same size can be multiplied together, and this is often called element-wise matrix multiplication. When referring to matrix multiplication, a different operator is often used, such as a circle “o”. As with element-wise subtraction and addition, element-wise multiplication involves the multiplication of elements from each parent matrix to calculate the values in the new matrix.

```
         a11 * b11, a12 * b12
A o B = (a21 * b21, a22 * b22)
         a31 * b31, a32 * b32
```



```python
# create two matrices do an element wise multiplication
M1 = np.array([[1, 2, 3], [4, 5, 6]])
M2 = np.array([[7, 8, 9], [10, 11, 12]])

M = M1 * M2

print(M1, '\no\n', M2, '\n=\n',M)
```

    [[1 2 3]
     [4 5 6]] 
    o
     [[ 7  8  9]
     [10 11 12]] 
    =
     [[ 7 16 27]
     [40 55 72]]


#### Division
A matrix can be divided by another matrix with the same dimensions. The scalar elements in the resulting matrix are calculated as the division of the elements in each of the matrices.
```

         a11 / b11, a12 / b12
A / B = (a21 / b21, a22 / b22)
         a31 / b31, a32 / b32
```


```python
# create two matrices divide one by the other 
M1 = np.array([[1, 2, 3], [4, 5, 6]])
M2 = np.array([[7, 8, 9], [10, 11, 12]])

M = M2 / M1

print(M2, '\n', '/', '\n', M1, '\n=\n',M)
```

    [[ 7  8  9]
     [10 11 12]] 
     / 
     [[1 2 3]
     [4 5 6]] 
    =
     [[7.  4.  3. ]
     [2.5 2.2 2. ]]


### Matrix-Matrix Dot-Product

We can calculate dot-products, as seen with vectors, to check the similarity between matrices. The matrix dot product is more complicated than the previous operations and involves a rule as not all matrices can be multiplied together.The rule is as follows:

>**The number of columns (n) in the first matrix (A) must equal the number of rows (m) in the second matrix (B).**

For example, think of a matrix A having m rows and n columns and matrix B having n rows and and k columns. Provided the n columns in A and n rows b are equal, the result is a new matrix with m rows and k columns.


>**C(m,k) = A(m,n) * B(n,k)**

The calculations are performed as shown below:
```
     a11, a12
A = (a21, a22)
     a31, a32
 
     b11, b12
B = (b21, b22)
 
     a11 * b11 + a12 * b21, a11 * b12 + a12 * b22
C = (a21 * b11 + a22 * b21, a21 * b12 + a22 * b22)
     a31 * b11 + a32 * b21, a31 * b12 + a32 * b22

```
This rule applies for a chain of matrix multiplications where the number of columns in one matrix in the chain must match the number of rows in the following matrix in the chain.
The intuition for the matrix multiplication is that we are calculating the dot product between each row in matrix A with each column in matrix B. For example, we can step down rows of column A and multiply each with column 1 in B to give the scalar values in column 1 of C.

This is made clear with the following image.

![](https://lh3.googleusercontent.com/Jmb3q-kvNTr1Lz3jgmIIxiPo_GGgwllP_FonFnSROmte0wc1KmM7d_aUJhHzPDsA7wB0es5OTHs51HSSlENTOltoa31TxzZMZlgGpf-l62gCHRaU3C_CsHUWw3orAOLCM5Lucpo)

Let's see how to achieve this in python and numpy with `.dot()` as we saw before:


```python
# matrix dot product
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[1, 2], [3, 4]])

C = A.dot(B)

print(A, '\n.', '\n', B, '\n=', C)
```

    [[1 2]
     [3 4]
     [5 6]] 
    . 
     [[1 2]
     [3 4]] 
    = [[ 7 10]
     [15 22]
     [23 34]]


So above , the output matrix has number of rows = number of rows in A , and number of columns = number of columns in B. This will always be the case for a matrix-matrix dot product. If we don't follow the rule stated above , python throws an error as shown below:


```python
# matrix dot product with mismatched dimensions - intended error
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[1, 2]])

C = A.dot(B)

print(A, '\n.', '\n', B, '\n=', C)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-10-1acc0dec94c8> in <module>()
          3 B = np.array([[1, 2]])
          4 
    ----> 5 C = A.dot(B)
          6 
          7 print(A, '\n.', '\n', B, '\n=', C)


    ValueError: shapes (3,2) and (1,2) not aligned: 2 (dim 1) != 1 (dim 0)


### Matrix-Vector Multiplication

A matrix and a vector can be multiplied together as long as the rule of matrix multiplication (stated above) is observed. The number of columns in the matrix must equal the number of rows in the vector. As with matrix multiplication, the operation can be written using the dot notation. Because the vector only has one column, the result is always a vector. See the general approach below where A is the matrix being multiplied to v, a vector
```
A = (a21, a22)
     a31, a32
 
     v1
v = (v2)
 
     a11 * v1 + a12 * v2
c = (a21 * v1 + a22 * v2)
     a31 * v1 + a32 * v2
```

The matrix-vector multiplication can be implemented in NumPy using the dot() function as seen before.


```python
# matrix-vector multiplication

A = np.array([[1, 2], [3, 4], [5, 6]])
v = np.array([0.5, 0.5])

C = A.dot(v)

print(A,'\ndot', '\n',v,'\n=',C)
```

    [[1 2]
     [3 4]
     [5 6]] 
    dot 
     [0.5 0.5] 
    = [1.5 3.5 5.5]


Similar to scalar-vector multiplication, a scalar-matrix multiplication involves multiplying every element of the matrix to the scalar value, resulting as an output matrix having same dimensions as the input matrix.  

### Matrix Transpose

Neural networks frequently process weights and inputs of different sizes where the dimensions do not meet the requirements of matrix multiplication. Matrix transpose provides a way to “rotate” one of the matrices so that the operation complies with multiplication requirements and can continue. There are two steps to transpose a matrix:

1. Rotate the matrix right 90° clockwise.
2. Reverse the order of elements in each row (e.g. [a b c] becomes [c b a]).

This can be better understood looking at this image : 
![](https://static1.squarespace.com/static/54ad91eae4b04d2abc8d6247/t/55f6238ee4b015f33a4d3b7a/1442194319173/)

Numpy provides the transpose operation by simply using `.T`  or `.np.transpose()` with the matrix that needs to be transposed as shown below:


```python
# create a transpose a matrix

a = np.array([
   [1, 2, 3], 
   [4, 5, 6],
   [7, 8, 9]])

a_transposed = a.T
a_transposed_2 = np.transpose(a)

print(a,'\n\n', a_transposed, '\n\n', a_transposed_2)

```

    [[1 2 3]
     [4 5 6]
     [7 8 9]] 
    
     [[1 4 7]
     [2 5 8]
     [3 6 9]] 
    
     [[1 4 7]
     [2 5 8]
     [3 6 9]]


## Summary

This lesson introduces how to create and process matrices using numpy and python. This lesson, along with the previous lesson of vectors, shows all the necessary operations (and more) required for solving the system of equation we saw earlier using matrix algebra. In the next lesson we shall apply the skills learnt so far on the apples and bananas problem that we defined at the very beginning. 
