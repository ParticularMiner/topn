import sys
import numpy as np

if sys.version_info[0] >= 3:
    from topn import topn as ct
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
    rr = r.astype(idx_dtype)
    len_r = len(r)
    if (n_rows + 1) > len_r:
        rr.resize(n_rows + 1, refcheck=False)
    cc = c.astype(idx_dtype)
    dd = d.copy()
    new_len = ct_thread.topn_threaded(
        rr, cc, dd,
        ntop,
        n_rows,
        n_jobs
    )
    
    rr.resize((new_len if n_rows < 0 else (n_rows + 1)), refcheck=False)
    cc.resize(new_len, refcheck=False)
    dd.resize(new_len, refcheck=False)
    return rr, cc, dd


def awesome_hstack_topn(blocks, ntop, use_threads=False, n_jobs=1):
    n_blocks = len(blocks)
    r = np.concatenate([b.indptr for b in blocks])
    c = np.concatenate([b.indices for b in blocks])
    d = np.concatenate([b.data for b in blocks])
    n_cols = np.array([b.shape[1] for b in blocks])
    M = blocks[0].shape[0]
    N = np.sum(n_cols)
    if len(d) > 0:
        hstack_indptr = np.empty(M + 1, dtype=c.dtype)
        hstack_indices = np.empty(len(c), dtype=c.dtype)
        hstack_data = np.empty(len(d), dtype=d.dtype)
        if not use_threads:
            ct.sparse_topn(
                n_blocks, M, n_cols,
                r, c, d,
                ntop,
                hstack_indptr, hstack_indices, hstack_data
            )
        else:
            ct_thread.sparse_topn_threaded(
                n_blocks, M, n_cols,
                r, c, d,
                ntop,
                hstack_indptr, hstack_indices, hstack_data,
                n_jobs
            )
    else:
        hstack_indptr = np.zeros(M + 1, dtype=c.dtype)
        hstack_indices = np.empty(0, dtype=c.dtype)
        hstack_data = np.empty(0, dtype=d.dtype)
    return csr_matrix((hstack_data, hstack_indices, hstack_indptr), shape=(M, N))


