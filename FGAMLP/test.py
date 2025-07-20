import torch
import numpy
from torch_sparse import SparseTensor

# nums=torch.load("./num_year.pt")
# print(f"len(nums)")
# for i in range(0,100):
#     print(f"year {i}: {nums[i]}")

a=SparseTensor(row=torch.tensor([1,2]),col=torch.tensor([4,5]), value=torch.tensor([1.000001,1.0001]).float())
# row side is target side
a=a.to_scipy(layout='csr')
x=numpy.ones((6,5))
b=a@x
print(dir(SparseTensor))
print(a)
print(b)

temp=torch.zeros((10))
for i in range(10):
    temp[i]=1.0
print(temp)

for i in range(10):
    temp[i]=(2.0/max(i,1))
print(temp)
