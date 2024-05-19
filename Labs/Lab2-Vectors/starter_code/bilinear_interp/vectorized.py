import numpy as np
from numpy import int64
def bilinear_interp_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This is the vectorized implementation of bilinear interpolation.
    - a is a ND array with shape [N, H1, W1, C], dtype = int64
    - b is a ND array with shape [N, H2, W2, 2], dtype = float64
    - return a ND array with shape [N, H2, W2, C], dtype = int64
    """
    # get axis size from ndarray shape
    N, H1, W1, C = a.shape
    N1, H2, W2, _ = b.shape
    assert N == N1
    # TODO: Implement vectorized bilinear interpolation
    res = np.empty((N, H2, W2, C), dtype=int64)
    # X and y values of all the target nodes from the matrix b
    x = b[...,0]
    y = b[...,1]
    #Integer part of x and y
    x_idx = np.floor(x).astype(int64)
    y_idx = np.floor(y).astype(int64)
    #Decimal part of x and y
    _x = x-x_idx
    _y = y-y_idx
    #Extenion of the n and _x, _y dimension to match x and y
    n_idx=np.arange(N)[:, None, None]
    _x = _x[...,None]
    _y = _y[...,None]
    #Get all the nearby points
    q11 = a[n_idx, x_idx, y_idx]
    q21 = a[n_idx, x_idx + 1, y_idx]
    q12 = a[n_idx, x_idx, y_idx + 1]
    q22 = a[n_idx, x_idx + 1, y_idx + 1]
    # Perform actual bilinear interpolation
    res = (q11 * (1 - _x) * (1 - _y) +
           q21 * _x * (1 - _y) +
           q12 * (1 - _x) * _y +
           q22 * _x * _y).astype(int64)

    
    return res
