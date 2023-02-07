import torch
print(torch.__version__)

torchdevice = torch.device('cpu')
if torch.cuda.is_available():
  torchdevice = torch.device('cuda')
  print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))
print('Running on ' + str(torchdevice))


import torch_sparse

# Covenience wrappers around torch_sparse matmult
def ts_spmm(A,B):
  Ats = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(A)
  AB = torch_sparse.matmul(Ats,B)
  return AB

print(f'GPU memory usage: {torch.cuda.memory_allocated(device=torchdevice)/10**9}')

# Dimension of the square sparse matrix
n = 50000
# Number of non-zero elements
nnz = 200
# Second dimension of the dense matrix
m = 200

rowidx = torch.randint(low=0, high=n, size=(nnz,), device=torchdevice)
colidx = torch.randint(low=0, high=n, size=(nnz,), device=torchdevice)
itemidx = torch.vstack((rowidx,colidx))
xvalues = torch.randn(nnz, device=torchdevice)

Y_dense = torch.randn((n,m), device=torchdevice)


# Require gradients on the values of the sparse matrix (not the matrix itself)
xvalues_0 = xvalues.detach().clone().requires_grad_(True)
X_sparse_0 = torch.sparse_coo_tensor(itemidx, xvalues_0, size=(n,n))
Y_dense_0 = Y_dense.detach().clone().requires_grad_(True)

xvalues_1 = xvalues.detach().clone().requires_grad_(True)
X_sparse_1 = torch.sparse_coo_tensor(itemidx, xvalues_1, size=(n,n))
Y_dense_1 = Y_dense.detach().clone().requires_grad_(True)

xvalues_2 = xvalues.detach().clone().requires_grad_(True)
X_sparse_2 = torch.sparse_coo_tensor(itemidx, xvalues_2, size=(n,n))
Y_dense_2 = Y_dense.detach().clone().requires_grad_(True)
gpu0 = torch.cuda.memory_allocated(device=None)/10**9
# print(f'GPU memory usage: {torch.cuda.memory_allocated(device=None)/10**9}')

# vanilla pytorch path
t1 = torch.sparse.mm(X_sparse_1, Y_dense_1).sum()
t1.backward()

gpu1 = torch.cuda.memory_allocated(device=None)/10**9
print(f'Vanilla torch SPMM, x.grad: {xvalues_1.grad}, y.grad: {Y_dense_1.grad}')
print(f'GPU memory usage: {gpu1-gpu0}')


# torch_sparse path
t0 = ts_spmm(X_sparse_0, Y_dense_0).sum()
t0.backward()

gpu2 = torch.cuda.memory_allocated(device=None)/10**9
print(f'torch_sparse SPMM, x.grad: {xvalues_0.grad}, y.grad: {Y_dense_0.grad}')
print(f'GPU memory usage: {gpu2-gpu1}')

t2 = torch.matmul(X_sparse_2.to_dense(), Y_dense_2).sum()
t2.backward()

gpu3 = torch.cuda.memory_allocated(device=None)/10**9
print(f'torch_sparse SPMM, x.grad: {xvalues_2.grad}, y.grad: {Y_dense_2.grad}')
print(f'GPU memory usage: {gpu3-gpu2}')
