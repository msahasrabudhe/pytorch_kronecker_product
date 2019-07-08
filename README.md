# pytorch_kronecker_product
Fast PyTorch implementation of the kronecker product for 2D matrices inside the autograd framework.


### Sample usage: 
```python
# Import the module defined in KroneckerProduct.py
from KroneckerProduct import KroneckerProduct

# Define shapes of matrices on which to compute the Kronecker product. 
# Shape is (batch_size, rows, columns)
A_shape = (1, 3, 5)
B_shape = (1, 2, 3)

# Initialise the KroneckerProduct module by specifying these shapes. 
# The batch size is to be excluded in this initialisation.
kronecker = KroneckerProduct(A_shape[1:], B_shape[1:])

# Initialise A and B randomly. 
A = torch.FloatTensor(*A_shape).random_() % 100
B = torch.FloatTensor(*B_shape).random_() % 100

# OPTIONAL: GPU usage. 
kronecker = kronecker.cuda()
A = A.cuda()
B = B.cuda()

# Compute Kronecker product. 
kprod = kronecker(A, B)
```


### Extensions
In the current version, the shapes of the two matrices must be specified while 
initialising the module. This can, however, be easily extended to a version which 
does not require specifying matrix sizes, albeit, at the cost of extra computation. 
