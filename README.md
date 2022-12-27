# Gaussian elimination in Python
## Preparation
1. Run the following command to install `pipenv` command.

    ```bash
    pip install pipenv
    ```

1. Execute the following command to install the dependent libraries.

    ```bash
    pipenv install
    ```

## Usage
Import `gaussian_elimination.py` in your python script and call `gaussian_elimination` function for following format.

```python
from gaussian_elimination import gaussian_elimination

# initialize data
ndim = 10
A = np.matrix(np.random.normal(3.1, 4.1, (ndim, ndim)), dtype=np.float64)
xexact = np.arange(ndim, dtype=np.float64) + 1
b = np.ravel(A @ xexact)

# call gaussian_elimination function
x = gaussian_elimination(A, b)
# output result
print('estimated: ', x)
```

## Example
Run the following command.

```bash
pipenv run python3 sample.py
```
