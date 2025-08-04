import torch
print(torch.version.cuda)             # None => no CUDA support in build
print(torch.cuda.is_available())  