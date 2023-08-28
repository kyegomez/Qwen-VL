[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Qwen-VL
My personal implementation of the model from "Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities", they haven't released model code yet sooo...


# Install
`pip3 install qwen`

---

# Usage
```python

import torch
from qwen.model import QwenVL

#usage
img = torch.randn(1, 3, 256, 256)
caption = torch.randint(0, 20000, (1, 1024))

model = QwenVL()
output = model(img, caption)
print(output.shape)

```

# Training


