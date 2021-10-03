# topn

Cython utility functions to be used instead of pandas' `SeriesGroupBy` `nlargest()` function (since [pandas does it so slowly](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.SeriesGroupBy.nlargest.html)).

Contains 3 functions:
1. `awesome_topn()`, 
2. `awesome_hstack_topn()`,
3. `awesome_hstack()`: (for CSR matrices only; at least twice as fast as `scipy.sparse.hstack` in scipy version 1.6.1)

See [Short Description](#desc) for details.


This is how it may be done with pandas:
```python
import pandas as pd
import numpy as np

r = np.array([0, 1, 2, 1, 2, 3, 2]) 
c = np.array([1, 1, 0, 3, 1, 2, 3]) 
d = np.array([0.3, 0.2, 0.1, 1.0, 0.9, 0.4, 0.6]) 
rcd = pd.DataFrame({'r': r, 'c': c, 'd': d})
rcd
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>r</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>1</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>2</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>3</td>
      <td>0.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
ntop = 2
```


```python
rcd.set_index('c').groupby('r')['d'].nlargest(ntop).reset_index().sort_values(['r', 'd'], ascending = [True, False])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>r</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>2</td>
      <td>0.4</td>
    </tr>
  </tbody>
</table>
</div>



## Usage
```python
from topn import awesome_topn

o_r, o_c, o_d = awesome_topn(r, c, d, ntop, n_jobs=7)
pd.DataFrame({'r': o_r, 'c': o_c, 'd': o_d})
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>r</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>2</td>
      <td>0.4</td>
    </tr>
  </tbody>
</table>
</div>

Alternatively, if one had a matrix encoding the above data:

```python
from scipy.sparse import csr_matrix 

csr = csr_matrix((d, (r, c)), shape=(4, 4))
```

then one could use the function `awesome_hstack_topn()` instead:
```python
from topn import awesome_hstack_topn 

topn_matrix = awesome_hstack_topn([csr], ntop=ntop)
o_r, o_c = topn_matrix.nonzero()
o_d = topn_matrix.data
pd.DataFrame({'r': o_r, 'c': o_c, 'd': o_d})
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>r</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>2</td>
      <td>0.4</td>
    </tr>
  </tbody>
</table>
</div>



## Short Description <a name="desc"></a>
Contains 3 functions:
1. `awesome_topn()`, 
2. `awesome_hstack_topn()`,
3. `awesome_hstack()`

```python
def awesome_topn(r, c, d, ntop, n_rows=-1, n_jobs=1):
    """
    r, c, and d are 1D numpy arrays all of the same length N. 
    This function will return arrays rn, cn, and dn of length n <= N such
    that the set of triples {(rn[i], cn[i], dn[i]) : 0 < i < n} is a subset of 
    {(r[j], c[j], d[j]) : 0 < j < N} and that for every distinct value 
    x = rn[i], dn[i] is among the first ntop existing largest d[j]'s whose 
    r[j] = x.

    Input:
        r and c: two 1D integer arrays of the same length
        d: 1D array of single or double precision floating point type of the
        same length as r or c
        ntop maximum number of maximum d's returned
        n_rows: an int. If > -1 it will replace output rn with Rn the
            index pointer array for the compressed sparse row (CSR) matrix
            whose elements are {C[rn[i], cn[i]] = dn: 0 < i < n}.  This matrix
            will have its number of rows = n_rows.  Thus the length of Rn is
            n_rows + 1
        n_jobs: number of threads, must be >= 1

    Output:
        (rn, cn, dn) where rn, cn, dn are all arrays as described above, or
        (Rn, cn, dn) where Rn is described above
        
    """


def awesome_hstack_topn(blocks, ntop, sort=True, use_threads=False, n_jobs=1):
    """
    Returns, in CSR format, the matrix formed by horizontally stacking the
    sequence of CSR matrices in parameter 'blocks', with only the largest ntop
    elements of each row returned.  Also, each row will be sorted in
    descending order only when 
        ntop < total number of columns in blocks or sort=True,
    otherwise the rows will be unsorted.
    
    :param blocks: list of CSR matrices to be stacked horizontally.
    :param ntop: int. The maximum number of elements to be returned for
        each row.
    :param sort: bool. Each row of the returned matrix will be sorted in
        descending order only when ntop < total number of columns in blocks
        or sort=True, otherwise the rows will be unsorted.
    :param use_threads: bool. Will use the multi-threaded versions of this
        routine if True otherwise the single threaded version will be used.
        In multi-core systems setting this to True can lead to acceleration.
    :param n_jobs: int. When use_threads=True, denotes the number of threads
        that are to be spawned by the multi-threaded routines. Recommended
        value is number of cores minus one.

    Output:
        (scipy.sparse.csr_matrix) matrix in CSR format 
    """


def awesome_hstack(blocks, use_threads=False, n_jobs=1):
    """
    Returns, in CSR format, the matrix formed by horizontally stacking the
    sequence of CSR matrices in parameter blocks.
    
    :param blocks: list of CSR matrices to be stacked horizontally.
    :param use_threads: bool. Will use the multi-threaded versions of this
        routine if True otherwise the single threaded version will be used.
        In multi-core systems setting this to True can lead to acceleration.
    :param n_jobs: int. When use_threads=True, denotes the number of threads
        that are to be spawned by the multi-threaded routines. Recommended
        value is number of cores minus one.

    Output:
        (scipy.sparse.csr_matrix) matrix in CSR format 
    """
```