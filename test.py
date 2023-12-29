import torch.nn.functional as F
import torch
def test():
    a  =  torch.rand([7,7],out=None)
    a0 = a.view(a.shape[0], a.shape[1], -1)
    a1 = a0.view(a0.shape[0], a0.shape[1],a0.shape[2],-1)

    a1 = a1.permute(2,3,0,1)
    b = torch.Tensor([[1,1,1],[0,0,0],[1,1,1]])

    b1 = b.view(b.shape[0], b.shape[1], -1)
    b1 = b1.view(b1.shape[0], b1.shape[1],b1.shape[2] ,-1)
    b1 = b1.permute(3, 2, 0, 1)
    a2 = F.conv_transpose2d(input=a1,weight=b1,padding=0,stride=1)
    print(a)
    print(a2.shape)


if __name__ == '__main__':
    test()