import torch

torch.manual_seed(42)
torch.ops.load_library("bld/libcudaexpr.so")

a = torch.randint(0, 10, (2, 3, 4), dtype=torch.float).cuda()
a_plus_one = torch.ops.cudaexpr.add_one(a)

print(a)
print(a_plus_one)
