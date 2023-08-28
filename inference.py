from qwen.inference import QwenVLChat


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