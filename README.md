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

## 📚 Training

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

