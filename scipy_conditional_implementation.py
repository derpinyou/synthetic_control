#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.optimize import fmin_slsqp
from functools import partial

from src.utils import timeit

def loss_w(W, X, y) -> float:
    return np.sqrt(np.mean((y - X.dot(W)) ** 2))

@timeit
def get_w(X, y):
    from functools import partial
    from scipy.optimize import minimize
    w_start = [1/X.shape[1]]*X.shape[1]
    print(X.shape, y.shape)
    weights = minimize(
        partial(loss_w, X=X, y=y),
        x0=w_start,
        bounds=[(0, 1)] * len(w_start),
        constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    )
    return weights.x

