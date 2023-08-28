To create a hackable class for QwenVLChat using Transformers, you can define a class with various options as parameters. Here's an example implementation:

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

class QwenVLChat:
    def __init__(
        self,
        model_name,
        device_map="cuda",
        trust_remote_code=True,
        bf16=False,
        fp16=False,
        cpu=False,
        seed=1234
    ):
        torch.manual_seed(seed)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        
        if bf16:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, trust_remote_code=trust_remote_code, bf16=True).eval()
        elif fp16:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, trust_remote_code=trust_remote_code, fp16=True).eval()
        elif cpu:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", trust_remote_code=trust_remote_code).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, trust_remote_code=trust_remote_code).eval()
        
        self.model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.history = None
    
    def chat(self, text):
        query = self.tokenizer.from_list_format(text)
        response, self.history = self.model.chat(self.tokenizer, query=query, history=self.history)
        
        return response
    
    def draw_bbox_on_latest_picture(self, response):
        image = self.tokenizer.draw_bbox_on_latest_picture(response, self.history)
        
        return image
You can then use the QwenVLChat class to perform inference using Qwen-VL-Chat or Qwen-VL. Here's an example usage:

# For Qwen-VL-Chat
qwen_chat = QwenVLChat(model_name="Qwen/Qwen-VL-Chat", device_map="cuda")
response = qwen_chat.chat([
    {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    {"text": "这是什么?"}
])
print(response)

response = qwen_chat.chat("框出图中击掌的位置")
print(response)

image = qwen_chat.draw_bbox_on_latest_picture(response)
if image:
    image.save("1.jpg")
else:
    print("no box")

# For Qwen-VL
qwen_vl = QwenVLChat(model_name="Qwen/Qwen-VL", device_map="cuda")
response = qwen_vl.chat([
    {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    {"text": "Generate the caption in English with grounding:"}
])
print(response)

image = qwen_vl.draw_bbox_on_latest_picture(response)
if image:
    image.save("2.jpg")
else:
    print("no box")
Remember to replace model_name with the correct model name for the specific Qwen-VL model you want to use.