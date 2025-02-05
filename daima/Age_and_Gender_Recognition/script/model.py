import torch.nn as nn
import torchvision.models as models

class AgeGenderModel(nn.Module):
    def __init__(self, num_age_classes=8):
        """
        使用 ResNet 的预训练模型进行年龄和性别分类
        :param num_age_classes: 年龄分类的类别数
        """
        super(AgeGenderModel, self).__init__()
        # 加载预训练的 ResNet
        self.base_model = models.resnet18(pretrained=True)  # 选择 ResNet18，可以更换为 ResNet50 等

        # 替换 ResNet 的最后一层
        num_features = self.base_model.fc.in_features  # 获取 ResNet 的最后一层输入特征数
        self.base_model.fc = nn.Identity()  # 去掉原始全连接层，保留特征提取部分

        # 性别分类分支（1 个输出）
        self.gender_fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # 二分类输出
        )

        # 年龄分类分支（num_age_classes 个输出）
        self.age_fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_age_classes),  # 多分类输出
        )

    def forward(self, x):
        # 使用 ResNet 提取特征
        features = self.base_model(x)

        # 性别和年龄分支的输出
        gender_output = self.gender_fc(features)
        age_output = self.age_fc(features)

        return gender_output, age_output
