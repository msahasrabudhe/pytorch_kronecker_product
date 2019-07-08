import torch
from torch import nn
from torch import optim
from KroneckerProduct import *
import sys
from torch.autograd import Variable

def write_flush(text, stream=sys.stdout):
    stream.write(text)
    stream.flush()

class Model(nn.Module):
    def __init__(self, A_shape, B_shape):
        super(Model, self).__init__()

        self.A_shape    = A_shape
        self.B_shape    = B_shape

        self.kronecker  = KroneckerProduct(A_shape[1:], B_shape[1:])
        self.register_parameter('A', nn.Parameter(torch.randint(100, size=A_shape).float()))

    def forward(self, B):
        return self.kronecker(self.A, B)

def main():
    batch_size      = 1
    A_shape         = (batch_size, 2, 2)
    B_shape         = (batch_size, 2, 3)

    kronecker       = KroneckerProduct(A_shape[1:], B_shape[1:])
    kronecker.cuda()

    A_target        = torch.randint(10, size=A_shape).float().cuda()
    B               = torch.randint(20, size=B_shape).float().cuda()
    C_target        = kronecker(A_target, B)

    model           = Model(A_shape, B_shape)
    model.cuda()

    optimiser       = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

    n_epochs        = 4000
    
    for epoch in range(n_epochs):
        optimiser.zero_grad()
        C           = model(B)

        loss        = torch.mean((C_target - C) ** 2)
        loss.backward()
        optimiser.step()
        
        write_flush('\r'+' '*100+'\rEpoch %d: Loss = %.4f' %(epoch, loss.item()))
        if epoch % (n_epochs//10) == 0:
            write_flush('\n')

    write_flush('\n')

    print('\nLearnt A: ')
    print(model.A)
    print('\nA_target:')
    print(A_target)
    print('\nB:')
    print(B)
    print('\nC_target: ')
    print(C_target)
    print('\nFinal C:')
    print(C)

        

if __name__ == '__main__':
    main()
