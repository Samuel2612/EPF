# -*- coding: utf-8 -*-
"""
Created on Thu May  1 10:07:40 2025

@author: samue
"""

import numpy as np
from itertools import product

def prod_cos(angles):
    """
    Direct product   ∏ cos(a_j).
    `angles` is any 1-D array-like of real numbers.
    """
    
    return np.prod(np.cos(angles))


def product_to_sum_identity(angles):
    r"""
    Implements the identity
    
        ∏_{j=1}^{n} cos(a_j)  =  (1/2^{n-1}) Σ_{s∈{1,±1,…}} cos(s·a),
    
    where the first sign in `s` is fixed to +1 (to avoid the ± redundancy
    coming from cos(−x)=cos(x)).
    """
    n = angles.size
    
    # generate all (n-1)-tuples of ±1; prepend 1 to each of them
    rest  = np.array(list(product((1, -1), repeat=n-1)), dtype=int)
    signs = np.column_stack((np.ones(rest.shape[0], dtype=int), rest))
    
    dots = signs @ angles              # every signed sum  s·a
    return np.sum(np.cos(dots)) / (2**(n-1))


# ------------ quick demonstration ----------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(12345)
    a = np.array(rng.uniform(0, 2*np.pi, size=5))   # five random angles
    
    lhs = prod_cos(a)                    # direct product
    rhs = product_to_sum_identity(a)     # right-hand side of the theorem
    
    print("∏ cos(a_j) =", lhs)
    print("average of cos(s·a) =", rhs)
