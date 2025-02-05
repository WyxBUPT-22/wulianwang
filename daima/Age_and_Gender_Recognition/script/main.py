import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import AgeGenderModel
from PIL import Image, ImageDraw, ImageFont
import asyncio
from facenet_pytorch import MTCNN
from collections import Counter
import time
import serial

ser = serial.Serial('COM7', 9600, timeout=1)
async def put_chinese_text(img, text, position, font_path="C:/Windows/Fonts/simsun.ttc", font_size=30,
                           color=(255, 0, 0)):
    """在图像上绘制中文文字"""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


async def capture_frame(cap):
    """捕获图像"""
    ret, frame = cap.read()
    return frame


async def process_frame(frame, model, device, mtcnn, transform, min_area, results_buffer):
    """处理图像并记录结果"""
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            w, h = x2 - x1, y2 - y1

            # 忽略小于阈值的人脸区域
            if w * h < min_area:
                continue

            # 提取人脸区域
            face_img = frame[y1:y2, x1:x2]
            if face_img is not None and face_img.size > 0:
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            else:
                continue

            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

            # 预处理人脸图像
            face_tensor = transform(face_pil)
            face_tensor = face_tensor.unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                pred_gender, pred_age = model(face_tensor)
                pred_gender = torch.sigmoid(pred_gender).item()
                pred_age = torch.argmax(pred_age, dim=1).item()

            # 性别解码
            gender_label = "男" if pred_gender > 0.5 else "女"

            # 年龄段解码（根据训练时的定义）
            age_labels = [
                "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
                "(38-43)", "(48-53)", "(60-100)"
            ]
            age_label = age_labels[pred_age]

            # 将当前结果加入缓冲区
            results_buffer['gender'].append(gender_label)
            results_buffer['age'].append(age_label)

            # 显示当前结果
            frame = await put_chinese_text(frame, f"性别: {gender_label}", (x1, y1 - 20), font_size=30, color=(255, 0, 0))
            frame = await put_chinese_text(frame, f"年龄: {age_label}", (x1, y1 - 50), font_size=30, color=(255, 0, 0))

    return frame


async def capture_and_infer(model, device):
    """异步摄像头捕获图像并进行推理"""
    cap = cv2.VideoCapture(0)

    # 初始化 MTCNN 人脸检测器
    mtcnn = MTCNN(keep_all=True, device=device)

    # 设置图像预处理转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 设置最小像素面积阈值
    min_area = 5000  # 仅处理像素面积大于此值的人脸

    # 初始化结果缓冲区
    results_buffer = {'gender': [], 'age': []}
    buffer_max_size = 70  # 每轮记录50次结果
    pause_duration = 7  # 暂停时长（秒）

    is_paused = False  # 标记是否在暂停
    pause_end_time = 0  # 暂停结束时间

    while True:
        frame = await capture_frame(cap)

        if frame is not None:
            current_time = time.time()

            if is_paused:
                # 暂停期间显示摄像头画面和最终结果
                frame = await put_chinese_text(frame, f"最终性别: {gender_most_common}", (50, 50), font_size=40,
                                               color=(0, 255, 0))
                frame = await put_chinese_text(frame, f"最终年龄: {age_most_common}", (50, 100), font_size=40,
                                               color=(0, 255, 0))
                if gender_most_common == "男" and age_most_common != "(0-2)" and age_most_common != "(4-6)" and age_most_common != "(8-12)":
                    if age_most_common != "(60-100)":
                        ser.write(b"1")  # 发送 1
                    else:
                        ser.write(b"3")  # 发送 3
                elif gender_most_common == "女" and age_most_common != "(0-2)" and age_most_common != "(4-6)" and age_most_common != "(8-12)":
                    if age_most_common != "(60-100)":
                        ser.write(b"2")  # 发送 2
                    else:
                        ser.write(b"3")  # 发送 3
                elif age_most_common == "(0-2)" or age_most_common == "(4-6)" or age_most_common == "(8-12)":
                    ser.write(b"0")  # 发送 0
                if current_time >= pause_end_time:
                    # 暂停结束，恢复识别
                    is_paused = False
            else:
                # 正常识别流程
                frame = await process_frame(frame, model, device, mtcnn, transform, min_area, results_buffer)


                if len(results_buffer['gender']) >= buffer_max_size:
                    gender_most_common = Counter(results_buffer['gender']).most_common(1)[0][0]
                    age_most_common = Counter(results_buffer['age']).most_common(1)[0][0]

                    # 清空缓冲区
                    results_buffer['gender'].clear()
                    results_buffer['age'].clear()

                    # 进入暂停
                    is_paused = True
                    pause_end_time = current_time + pause_duration

            # 显示画面
            cv2.imshow('Face Detection & Prediction', frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ser.close()  # 关闭串口


# 主函数
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = AgeGenderModel()
    model.load_state_dict(torch.load("age_gender_model_epoch_10.pth", map_location=device))
    model.to(device)
    model.eval()

    # 启动异步处理
    asyncio.run(capture_and_infer(model, device))
