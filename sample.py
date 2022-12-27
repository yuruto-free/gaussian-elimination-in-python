import numpy as np
from gaussian_elimination import gaussian_elimination

if __name__ == '__main__':
    ndim = 10
    A = np.matrix(np.random.normal(3.1, 4.1, (ndim, ndim)), dtype=np.float64)
    xexact = np.arange(ndim, dtype=np.float64) + 1
    b = np.ravel(A @ xexact)

    x = gaussian_elimination(A, b)
    print('estimated: ', x)
