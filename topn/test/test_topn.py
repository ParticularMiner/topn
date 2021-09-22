import unittest
import pandas as pd
import numpy as np
from scipy.sparse.csr import csr_matrix
from topn import awesome_topn
from unittest.mock import patch

class topnTest(unittest.TestCase):
    def test_topn(self):
        (r, c, d) = (
            np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6], dtype=np.int64),
            np.array([0, 1, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 0, 5, 3, 4, 3, 4, 3, 4, 3, 4, 4, 3, 5], dtype=np.int64),
            np.array([0.99999994, 0.25998905, 1.0000001 , 0.6412701 , 0.25998905, 1.        , 0.6412701 , 1.        , 0.6412701 , 1.        , 0.6412701 , 0.33135417, 0.21248752, 0.808251  , 0.808251  , 0.6412701 , 0.21248752, 1.        , 0.33135417, 1.        , 0.33135417, 1.        , 0.33135417, 1.0000001 , 0.33135417, 0.9999999 ], dtype=np.float32)
        )
        # print('')
        # print(f'lengths_in = {(len(r), len(c), len(d))}')
        ntop = 6
        n_rows = 6
        n_jobs = 7
        r, c, d = awesome_topn(r, c, d, ntop, n_rows, n_jobs)
        # print(f'(r, c, d) = {(r, c, d)}', flush=True)
        # print(f'lengths_out = {(len(r), len(c), len(d))}')
       
        
if __name__ == '__main__':
    unittest.main()
