import torch
print(torch.cuda.is_available())  # 确认是否为True
print(torch.cuda.current_device())  # 查看当前使用的GPU
print(torch.cuda.get_device_name(0))  # 查看GPU名称
