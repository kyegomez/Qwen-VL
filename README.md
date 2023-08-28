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

----

# Inference
```python

from qwen.inference import QwenVLChat


qwen_chat = QwenVLChat(model_name="Qwen/Qwen-VL-Chat", device_map="cuda")
response = qwen_chat.chat([
    {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    {"text": "这是什么?"}
])
print(response)



```


# Training
* [There is a file with a table of all the datasets used in the paper here](docs/datasets.md)

```python
from qwen.train import Train


def train():
    os.environ['MASTER_ADDR'] #'localhost'
    os.environ['MASTER_PORT'] #= '9994'
    
    # # [CRITICAL] Pay attention to this when scaling to multiple GPUs and clusters
    os.environ['RANK']       #= str(0) # Number of nodes (servers)
    os.environ['WORLD_SIZE'] # = str(torch.cuda.device_count())

    dist.init_process_group(backend='nccl') #init_method="env://")
    
    Train()

if __name__ == '__main__':
    train()


```

1. Set the environment variables:
   - `ENTITY_NAME`: Your wandb project name
   - `OUTPUT_DIR`: Directory to save the weights (e.g., `./weights`)
   - `MASTER_ADDR`: For distributed training
   - `MASTER_PORT` For master port distributed training
   - `RANK`- Number of nodes services
   - `WORLD_SIZE` Number of gpus

2. Configure the training:
   - Accelerate Config
   - Enable Deepspeed 3
   - Accelerate launch train_distributed_accelerate.py

For more information, refer to the [Training SOP](DOCs/TRAINING.md).


----



# Todo

- [ ] Position aware vision language adapter, compresses image features. Singer layer cross attention module inited randomly => group of trainable embeddings as query vectors + image features from the visual encoder as keys for cross attention ops => OUTPUT: compresses visual feature sequence to a fixed lnegth of 256, 2d absolute positional encodings are integrated into the cross attentions mechanisms query key pairs => compressed feature sequence of length of 256 => fed into decoder llm

- [ ] Bounding Boxes, for any given accurate bounding box, a norm process is applied in the range [0, 1000] and transformed into a string format (Xtope, Ytople)(Xottomright, Ybottomright) -> the string is tokenized as text and does not require positional vocabulary. Detection strings and regular text strings, two special tokens <box> and </box> are added to the beginning and end of the bounding box string. + another sed of special tokens (<ref> and </ref>) is introduced.

# Citations

Please use the following to cite this work:

```latex
@article{bai2023qwen,
  title={Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023},
  url={https://doi.org/10.48550/arXiv.2308.12966}
}

```

For more details, please refer to the [full paper](https://doi.org/10.48550/arXiv.2308.12966).


