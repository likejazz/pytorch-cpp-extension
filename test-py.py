import torch
import pycudaexpr

torch.manual_seed(42)

a = torch.randint(0, 10, (2, 3, 4), dtype=torch.float).cuda()
a_plus_one = pycudaexpr.add_one(a)

print(a)
print(a_plus_one)
