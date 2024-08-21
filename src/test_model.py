import torch
from model import transformer_builder

device = "cuda" if torch.cuda.is_available() else "cpu"
transformer = transformer_builder(10,10,30).to(device)

pad = torch.tensor([0], dtype=torch.int64)

# (B x Seq x d_model)
mask = torch.zeros()
src  = torch.rand([4,5,10], dtype=torch.int64)

transformer.encode(src, )
