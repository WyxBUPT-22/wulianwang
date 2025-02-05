import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import AgeGenderModel  # 导入定义的模型


# 定义推理函数
def infer(image_path, model_path, device):
    """
    对单张图片进行推理。
    :param image_path: 图片路径
    :param model_path: 训练好的模型权重文件路径
    :param device: 设备（'cpu' 或 'cuda'）
    :return: 性别和年龄预测
    """
    # 加载模型
    model = AgeGenderModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载图片并进行预处理
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度

    # 推理
    with torch.no_grad():
        pred_gender, pred_age = model(image_tensor)
        pred_gender = torch.sigmoid(pred_gender).item()  # 性别概率（0 到 1）
        pred_age = torch.argmax(pred_age, dim=1).item()  # 年龄类别

    # 性别解码
    gender_label = "Male" if pred_gender > 0.5 else "Female"

    # 年龄段解码（根据训练时的定义）
    age_labels = [
        "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
        "(38-43)", "(48-53)", "(60-100)"
    ]
    age_label = age_labels[pred_age]

    return gender_label, age_label


# 主函数
if __name__ == "__main__":
    # 输入图片路径和模型路径
    image_path = r"E:\_MyCollegeLife\_3rd_1\wulianwang\daima\cctest\微信图片_20250104150213.png"  # 替换为测试图片路径
    model_path = "age_gender_model_epoch_10.pth"  # 替换为保存的模型路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 推理
    gender, age = infer(image_path, model_path, device)

    # 打印结果
    print(f"Predicted Gender: {gender}")
    print(f"Predicted Age Range: {age}")
