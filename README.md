# topn

Utility function for `string_grouper` to use instead of pandas' `SeriesGroupBy` `nlargest()` function (since [pandas does it so slowly](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.SeriesGroupBy.nlargest.html)).

```python
import pandas as pd
import numpy as np

r = np.array([0, 1, 2, 1, 2, 3, 2]) 
c = np.array([1, 1, 0, 3, 1, 2, 3]) 
d = np.array([0.0, 0.2, 0.1, 1.0, 0.9, 0.4, 0.6]) 
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
      <td>0.0</td>
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
      <td>0.0</td>
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

r, c, d = awesome_topn(r, c, d, ntop, n_jobs=7)
pd.DataFrame({'r': r, 'c': c, 'd': d})
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
      <td>0.0</td>
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

## Short Description

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
        (Rn, cn, dn) when n_rows > -1, where Rn is described above 
        
    """
```