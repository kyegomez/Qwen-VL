import torch
from qwen.model import QwenVL

# usage
img = torch.randn(1, 3, 256, 256)
caption = torch.randint(0, 20000, (1, 1024))

model = QwenVL()
output = model(img, caption)
print(output.shape)
