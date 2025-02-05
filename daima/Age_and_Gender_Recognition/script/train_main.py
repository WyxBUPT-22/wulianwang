from train_model import train_model  # 从 train_model.py 导入主训练函数

if __name__ == "__main__":
    # 数据路径
    label_file = "../all_data.txt"  # 标签文件路径
    image_dir = "../aligned"  # 图片根目录

    # 训练模型
    train_model(label_file, image_dir, num_epochs=10, batch_size=32, learning_rate=0.0001)