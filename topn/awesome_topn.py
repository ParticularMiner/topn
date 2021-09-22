import sys
import numpy as np

if sys.version_info[0] >= 3:
    # from topn import topn as ct
    from topn import topn_threaded as ct_thread
else:
    # import topn as ct
    import topn_threaded as ct_thread


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
    dtype = r.dtype
    assert c.dtype == dtype

    idx_dtype = np.int32
    rr = np.asarray(r.copy(), dtype=idx_dtype)
    len_r = len(r)
    if (n_rows + 1) > len_r:
        np.resize(rr, n_rows + 1)
    cc = np.asarray(c.copy(), dtype=idx_dtype)
    dd = d.copy()
    new_len = ct_thread.topn_threaded(
        rr, cc, dd,
        ntop,
        n_rows,
        n_jobs
    )
    
    rr.resize((new_len if n_rows < 0 else (n_rows + 1)))
    cc.resize(new_len)
    dd.resize(new_len)
    return rr, cc, dd
