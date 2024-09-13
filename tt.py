import torch

a = torch.randint(0, 10, (5,))
b = torch.randint(0, 10, (5,))
c = torch.max(a, b)
pass