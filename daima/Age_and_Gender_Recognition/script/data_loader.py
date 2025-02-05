import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AgeGenderDataset(Dataset):
    def __init__(self, data, image_dir, transform=None):
        """
        初始化数据集
        :param data: DataFrame，包含标签和路径信息
        :param image_dir: 图片目录路径
        :param transform: 数据预处理或增强方法
        """
        self.data = data
        self.image_dir = os.path.abspath(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        gender = row['gender']
        age_category = row['age_category']

        # 加载图片并应用预处理
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 转换性别标签
        gender = 1 if gender == 'm' else 0

        return image, gender, age_category

def parse_and_prepare_data(label_file, image_dir):
    """
    解析标签文件并准备数据。
    :param label_file: 标签文件路径
    :param image_dir: 图片目录路径
    :return: 数据 DataFrame
    """
    # 加载标签文件
    data = pd.read_csv(label_file, sep="\t")

    # 定义年龄范围到类别的映射
    age_mapping = {
        '(0, 2)': 0,
        '(4, 6)': 1,
        '(8, 12)': 2,
        '(15, 20)': 3,
        '(25, 32)': 4,
        '(38, 43)': 5,
        '(48, 53)': 6,
        '(60, 100)': 7
    }

    # 将年龄映射为类别
    def map_age(age):
        if pd.isna(age):
            return -1
        if age in age_mapping:
            return age_mapping[age]
        try:
            age = int(age)
            if 0 <= age <= 2:
                return 0
            elif 3 <= age <= 6:
                return 1
            elif 7 <= age <= 12:
                return 2
            elif 13 <= age <= 20:
                return 3
            elif 21 <= age <= 32:
                return 4
            elif 33 <= age <= 43:
                return 5
            elif 44 <= age <= 53:
                return 6
            elif age >= 60:
                return 7
            else:
                return -1
        except ValueError:
            return -1

    data['age_category'] = data['age'].apply(map_age)

    # 拼接图片路径
    data['image_path'] = data.apply(
        lambda row: os.path.join(
            os.path.abspath(image_dir),
            row['user_id'],
            f"landmark_aligned_face.{row['face_id']}.{row['original_image']}"
        ),
        axis=1
    )

    # 过滤掉无效数据
    data = data[
        (data['image_path'].apply(os.path.exists)) &
        (data['age_category'] != -1)
    ].reset_index(drop=True)

    print(f"Dataset size after filtering: {len(data)}")
    return data

def get_dataloaders(label_file, image_dir, batch_size=32, test_size=0.2, random_state=42):
    """
    创建训练集和测试集的数据加载器。
    :param label_file: 标签文件路径
    :param image_dir: 图片所在目录路径
    :param batch_size: 批大小
    :param test_size: 测试集占比
    :param random_state: 随机种子
    :return: 训练和测试数据加载器
    """
    # 加载数据并解析
    data = parse_and_prepare_data(label_file, image_dir)

    # 数据划分
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    # 定义数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.CenterCrop(224),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集
    train_dataset = AgeGenderDataset(train_data, image_dir, transform=train_transform)
    test_dataset = AgeGenderDataset(test_data, image_dir, transform=test_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader
