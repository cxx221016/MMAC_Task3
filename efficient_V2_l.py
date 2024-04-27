'''
import os
import cv2
import torch
from torch import nn
import timm
from torchvision import transforms

class EfficientNetV2L(nn.Module):
    def __init__(self, out_size=1, model_name='tf_efficientnetv2_l'):
        super(EfficientNetV2L, self).__init__()
        # Load a pre-trained EfficientNet V2 Large
        self.efficientnet = timm.create_model(model_name, pretrained=True)
        # Replace the classifier with a new one for our specific task
        feature_dim = self.efficientnet.get_classifier().in_features
        self.efficientnet.classifier = nn.Linear(feature_dim, out_size)

    def forward(self, x):
        x = self.efficientnet(x)
        return x

class Model:
    def __init__(self):
        self.checkpoint = ""
        self.device = torch.device("cpu")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.386, 0.186, 0.024], [0.241, 0.125, 0.049])])
        
    def load(self, dir_path):
        self.model = EfficientNetV2L(out_size=1)  # Use the new EfficientNetV2L model
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_image, patient_info_dict):
        image = cv2.resize(input_image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device, torch.float)
        
        with torch.no_grad():
            score = self.model(image)

        score = score.detach().cpu()
        return float(score)

if __name__ == '__main__':
    net = EfficientNetV2L(out_size=1)
    x = torch.rand((32,3,512,512))
    y = net(x)
    print(y.shape)
'''

import os
import cv2
import torch
from torch import nn
import timm
from torchvision import transforms

class EfficientNetV2L(nn.Module):
    def __init__(self, num_additional_features=4, out_size=1, model_name='tf_efficientnetv2_l'):
        super(EfficientNetV2L, self).__init__()
        # 加载预训练的EfficientNet
        self.efficientnet = timm.create_model(model_name, pretrained=True)
        # 获取EfficientNet的分类器的输入特征数
        feature_dim = self.efficientnet.get_classifier().in_features
        
        # 替换EfficientNet的分类器
        self.efficientnet.classifier = nn.Identity()  # Remove original classifier

        # 针对非图像数据创建前馈层
        self.additional_fc = nn.Linear(num_additional_features, 32)
        
        # 创建新的分类器，它结合了图像特征和非图像特征
        self.classifier = nn.Linear(feature_dim + 32, out_size)
        
    def forward(self, image, additional_data):
        # 处理图像特征
        print(image.shape)  
        image_features = self.efficientnet(image)
        
        # 处理非图像特征
        additional_features = self.additional_fc(additional_data)
        
        # 合并图像特征和非图像特征
        combined_features = torch.cat((image_features, additional_features), dim=1)
        
        # 通过新的分类器进行预测
        output = self.classifier(combined_features)
        return output

class Model:
    def __init__(self):
        self.checkpoint = ""
        self.device = torch.device("cpu")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.386, 0.186, 0.024], [0.241, 0.125, 0.049])])
        
    def load(self, dir_path):
        self.model = EfficientNetV2L(num_additional_features=4, out_size=1)
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_image, patient_info):
        image = cv2.resize(input_image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device, torch.float)
        
        if not isinstance(patient_info, torch.Tensor):
            patient_info = torch.tensor(patient_info, dtype=torch.float32)
        patient_info = patient_info.to(self.device).unsqueeze(0)  # 确保patient_info是batch形式

        with torch.no_grad():
            score = self.model(image, patient_info)

        score = score.detach().cpu()
        return float(score)

if __name__ == '__main__':
    net = EfficientNetV2L(out_size=1)
    image = torch.rand((32,3,512,512))
    patient_info = torch.rand((32,4))
    y = net(image, patient_info)
    print(y.shape)