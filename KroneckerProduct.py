import torch 
import torch.nn as nn

class KroneckerProduct(nn.Module):
    """
    Class to perform the Kronecker Product inside the Autograd framework.

    See: https://en.wikipedia.org/wiki/Kronecker_product

    Usage: 
      * Initialise an instance of this class by specifying the shapes of A and B. 
      * OPTIONAL: If A and B are supposed to be cuda tensors, .cuda() for this class
        can be called. 
      * Call the class on A and B, which calls the forward function.
    """
    def __init__(self, A_shape, B_shape):
        """
        Inputs: 
            A_shape         A tuple of length 2 specifying the shape of matrix A
            B_shape         A tuple of length 2 specifying the shape of matrix B
        """

        super(KroneckerProduct, self).__init__()

        # Extract rows and columns. 
        Ar, Ac              = A_shape
        Br, Bc              = B_shape

        # Output size. 
        Fr, Fc              = Ar * Br, Ac * Bc
   
        # Shape for the left-multiplication matrix
        left_mat_shape      = (Fr, Ar)
        # Shape for the right-multiplication matrix. 
        right_mat_shape     = (Ac, Fc)
  
        ratio_left          = Ar
        ratio_right         = Ac
   
        # Initialise left- and right-multiplication matrices. 
        self.left_mat       = torch.FloatTensor(*left_mat_shape).fill_(0)
        self.right_mat      = torch.FloatTensor(*right_mat_shape).fill_(0)
   
        # Set 1s at the appropriate locations for the left-multiplication matrix. 
        for i in range(ratio_left):
            sr              = i * Br
            er              = sr + Br
    
            self.left_mat[sr:er, i]  = 1
        
        # Set 1s at the appropriate locations for the right-multiplication matrix. 
        for j in range(ratio_right):
            sc              = j * Bc
            ec              = sc + Bc
            self.right_mat[j, sc:ec] = 1

        # Function to expand A as required by the Kronecker Product. 
        self.A_expander     = lambda A: torch.mm(self.left_mat, torch.mm(A, self.right_mat))

        # Function to tile B as required by the kronecker product. 
        self.B_tiler        = lambda B: B.repeat([Ar, Ac])

    def cuda(self):
        """
        Override the native cuda method.
        """
        self.left_mat       = self.left_mat.cuda()
        self.right_mat      = self.right_mat.cuda()

    def forward(self, A, B):
        """
        Compute the Kronecker product for A and B. 
        """
        # This operation is a simple elementwise-multiplication of the expanded and tiled matrices. 
        return self.A_expander(A) * self.B_tiler(B)



