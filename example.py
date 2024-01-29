# Importing the necessary libraries
import torch
from qwen import Qwen

# Creating an instance of the Qwen model
model = Qwen()

# Generating random text and image tensors
text = torch.randint(0, 20000, (1, 1024))
img = torch.randn(1, 3, 256, 256)

# Passing the image and text tensors through the model
out = model(img, text)  # (1, 1024, 20000)
