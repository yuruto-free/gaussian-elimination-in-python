import numpy as np

def gaussian_elimination(_A, _b, pmeps=1e-10):
    """
    Parameters
    ----------
    _A : numpy.matrix
        coefficient matrix
    _b : numpy.ndarray
        right-hand side vector
    pmeps : double (option)
        pseudo machine epsilon

    Returns
    -------
    x : numpy.ndarray
        estimated vector
    """
    A = _A.copy()
    b = _b.copy()
    ndim = A.shape[-1]

    for idx in np.arange(ndim - 1):
        # search maximum value in idx-th column
        pos = np.abs(A[idx:, idx]).argmax()

        if pos != 0:
            # calculate the pivot
            pivot = pos + idx
            # swap
            A[idx, :], A[pivot, :] = A[pivot, :].copy(), A[idx, :].copy()
            b[idx], b[pivot] = b[pivot], b[idx]

        diag = A[idx, idx]
        if (np.abs(diag) < pmeps):
            raise Exception(f'Error: diagonal component is less than pseudo machine eps={pmeps}')

        # =============
        # forward erase
        # =============
        for row in np.arange(idx + 1, ndim):
            scale = A[row, idx] / diag
            A[row, idx:] -= scale * A[idx, idx:]
            b[row] -= scale * b[idx]

    # =================
    # back substitution
    # =================
    b[-1] /= A[-1, -1]
    for row in np.arange(ndim - 1)[::-1]:
        b[row] = (b[row] - np.dot(A[row, (row+1):], b[(row+1):])) / A[row, row]

    x = b.copy()

    return x

