from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

class QwenVLChat:
    def __init__(self,
                 model_name,
                 device_map="cuda",
                 trust_remote_code=True,
                 bf16=False,
                 fp16=False,
                 cpu=False,
                 seed=1234):
        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                       trust_remote_code=trust_remote_code)

        if bf16:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, 
                                                              trust_remote_code=trust_remote_code, bf16=True).eval()
        elif fp16:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, 
                                                              trust_remote_code=trust_remote_code, fp16=True).eval()
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
    
    def draw_box(self, response):
        image = self.tokenizer.draw_bbox_on_latest_picture(response, self.history)
        return image
    