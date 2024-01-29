[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Qwen-VL
My personal implementation of the model from "Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities", they haven't released model code yet sooo... 
For more details, please refer to the [full paper](https://doi.org/10.48550/arXiv.2308.12966).


# Install
`pip3 install qwen`

---

# Usage
```python

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

```

# Todo

- [ ] Position aware vision language adapter, compresses image features. Singer layer cross attention module inited randomly => group of trainable embeddings as query vectors + image features from the visual encoder as keys for cross attention ops => OUTPUT: compresses visual feature sequence to a fixed lnegth of 256, 2d absolute positional encodings are integrated into the cross attentions mechanisms query key pairs => compressed feature sequence of length of 256 => fed into decoder llm

- [ ] Bounding Boxes, for any given accurate bounding box, a norm process is applied in the range [0, 1000] and transformed into a string format (Xtope, Ytople)(Xottomright, Ybottomright) -> the string is tokenized as text and does not require positional vocabulary. Detection strings and regular text strings, two special tokens <box> and </box> are added to the beginning and end of the bounding box string. + another sed of special tokens (<ref> and </ref>) is introduced.

# Citations

Please use the following to cite this work:

```bibtex
@article{bai2023qwen,
  title={Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023},
  url={https://doi.org/10.48550/arXiv.2308.12966}
}

```