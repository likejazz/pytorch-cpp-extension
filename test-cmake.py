import torch
import time

torch.manual_seed(42)
torch.ops.load_library("bld/libcudaexpr.so")

a = torch.randint(0, 10, (2, 3, 4), dtype=torch.float).cuda()
start_time = time.time()
a_plus_one = torch.ops.cudaexpr.add_one(a)
print(f'{(time.time() - start_time):.4f}s elapsed.')

print(a)
print(a_plus_one)
