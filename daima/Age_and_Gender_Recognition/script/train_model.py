import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from model import AgeGenderModel  # 自定义的模型类
from tqdm import tqdm
import os


# 训练一个周期
def train_epoch(model, dataloader, criterion_gender, criterion_age, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_gender = 0
    correct_age = 0
    total = 0

    # 直接用 tqdm 包裹 dataloader
    loop = tqdm(dataloader, desc="Training", leave=True)  # leave=True 保持进度条显示

    for images, genders, ages in loop:  # 直接迭代 tqdm 对象
        images, genders, ages = images.to(device), genders.to(device), ages.to(device)

        # 前向传播
        pred_gender, pred_age = model(images)
        loss_gender = criterion_gender(pred_gender.squeeze(), genders.float())
        loss_age = criterion_age(pred_age, ages)
        loss = loss_gender + loss_age

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        correct_gender += ((torch.sigmoid(pred_gender.squeeze()) > 0.5) == genders).sum().item()
        correct_age += (pred_age.argmax(dim=1) == ages).sum().item()
        total += genders.size(0)

        # 更新 tqdm 显示信息
        loop.set_postfix(loss=loss.item(), gender_acc=correct_gender / total, age_acc=correct_age / total)

    return running_loss / len(dataloader), correct_gender / total, correct_age / total

# 验证一个周期

def validate_epoch(model, dataloader, criterion_gender, criterion_age, device):
    model.eval()
    running_loss = 0.0
    correct_gender = 0
    correct_age = 0
    total = 0

    # 使用 tqdm 显示验证进度条
    loop = tqdm(dataloader, total=len(dataloader), desc="Validating")

    with torch.no_grad():
        for images, genders, ages in loop:
            images, genders, ages = images.to(device), genders.to(device), ages.to(device)

            # 前向传播
            pred_gender, pred_age = model(images)
            loss_gender = criterion_gender(pred_gender.squeeze(), genders.float())
            loss_age = criterion_age(pred_age, ages)
            loss = loss_gender + loss_age

            # 统计损失和准确率
            running_loss += loss.item()
            correct_gender += ((torch.sigmoid(pred_gender.squeeze()) > 0.5) == genders).sum().item()
            correct_age += (pred_age.argmax(dim=1) == ages).sum().item()
            total += genders.size(0)

            # 更新 tqdm 显示
            loop.set_postfix(loss=loss.item(), gender_acc=correct_gender / total, age_acc=correct_age / total)

    return running_loss / len(dataloader), correct_gender / total, correct_age / total


# 主训练函数
from torch.utils.tensorboard import SummaryWriter

def train_model(label_file, image_dir, num_epochs=10, batch_size=32, learning_rate=0.0001):
    # 加载数据集
    label_file = "../all_data.txt"  # 替换为标签文件路径
    image_dir = "../aligned"  # 替换为图片目录路径

    # 获取数据加载器
    train_loader, test_loader = get_dataloaders(label_file, image_dir,32,0.2,42)

    # 模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AgeGenderModel(num_age_classes=8).to(device)
    criterion_gender = nn.BCEWithLogitsLoss()  # 二分类损失
    criterion_age = nn.CrossEntropyLoss()  # 多分类损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 初始化 TensorBoard
    log_dir = os.path.join(os.getcwd(), "logs")  # 使用绝对路径
    writer = SummaryWriter(log_dir=log_dir)

    # 添加模型结构到 TensorBoard
    sample_images, _, _ = next(iter(train_loader))
    writer.add_graph(model, sample_images.to(device))

    for epoch in range(num_epochs):
        # 训练和验证
        train_loss, train_gender_acc, train_age_acc = train_epoch(
            model, train_loader, criterion_gender, criterion_age, optimizer, device
        )
        val_loss, val_gender_acc, val_age_acc = validate_epoch(
            model, test_loader, criterion_gender, criterion_age, device
        )

        # 学习率调整
        scheduler.step()

        # 打印结果
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Gender Acc: {train_gender_acc:.4f}, Age Acc: {train_age_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Gender Acc: {val_gender_acc:.4f}, Age Acc: {val_age_acc:.4f}")

        # 记录到 TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("Accuracy/Gender/train", train_gender_acc, epoch + 1)
        writer.add_scalar("Accuracy/Gender/val", val_gender_acc, epoch + 1)
        writer.add_scalar("Accuracy/Age/train", train_age_acc, epoch + 1)
        writer.add_scalar("Accuracy/Age/val", val_age_acc, epoch + 1)

        # 保存模型
        torch.save(model.state_dict(), f"age_gender_model_epoch_{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")

    # 关闭 TensorBoard
    writer.close()
if __name__ == "__main__":
    # 文件路径
    label_file = "../all_data.txt"  # 替换为实际标签文件路径
    image_dir = "../aligned"  # 替换为实际图片文件路径
    # 训练模型
    train_model(label_file, image_dir, num_epochs=10, batch_size=32, learning_rate=0.0001)

