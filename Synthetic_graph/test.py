import torch

# x=torch.randn(4)
# z=torch.randn(4)
# print(x<0)
# print(z<0)
# print(((x<0)*(z<0)).nonzero())
# idx=torch.squeeze(((x<0)*(z<0)).nonzero())
# print(x)
# print(idx)
# y=torch.randn(4,5)
# print(y)
# print(y[idx])
# y=torch.mean(y[idx],dim=0)
# print(y)

x=torch.randn(10000,5)
covx=torch.cov(x.transpose(0,1))
print(covx.shape)
print("Covx : ",covx)
print(x)

print("Diagonalize : ")
#print(torch.diag(covx))

#print("Other method : ")
L,Q=torch.linalg.eigh(covx)
print(L,Q)
L=L.real
Lsq=torch.sqrt(L)
Isq=1/Lsq
print(L, Lsq,Isq)
print(torch.diag(Lsq)@torch.diag(Isq))

Q=Q.real
#print(torch.diag(L))
print(Q@Q.transpose(0,1))
print(torch.dist(Q@torch.diag(L)@Q.T,covx))

for i in range(len(x)):
    x[i]=torch.diag(Isq)@Q.T@x[i]
print(x.shape)
print(torch.cov(x.T))