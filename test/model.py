import os
import cv2
import torch
from torch import nn
import timm
from torchvision import transforms
import ttach as tta

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

class model:
    def __init__(self):
        self.checkpoint = "efficientnet1.pth"
        self.checkpoint1 = "efficientnet2.pth"
        self.device = torch.device("cuda:0")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.386, 0.186, 0.024], [0.241, 0.125, 0.049])])
        
    def load(self, dir_path="/home/shuochen/MMAC_Task3/test"):
        self.model = EfficientNetV2L(out_size=1)  # Use the new EfficientNetV2L model
        self.model1 = EfficientNetV2L(out_size=1)  # Use the new EfficientNetV2L model
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        checkpoint_path1 = os.path.join(dir_path, self.checkpoint1)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model1.load_state_dict(torch.load(checkpoint_path1, map_location=self.device))
        self.model.to(self.device)
        self.model1.to(self.device)
        self.model.eval()
        self.model1.eval()
        self.model = tta.ClassificationTTAWrapper(self.model, tta.aliases.d4_transform(), merge_mode='mean')
        self.model1 = tta.ClassificationTTAWrapper(self.model1, tta.aliases.d4_transform(), merge_mode='mean')


    def predict(self, input_image):
        image = cv2.resize(input_image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device, torch.float)
        
        with torch.no_grad():
            score = self.model(image)
            score1 = self.model1(image)
            score = (score + score1) / 2.0

        return score.detach().cpu().numpy()[0]

if __name__ == "__main__":
    model = model()
    model.load()

    image_paths = ["/home/shuochen/MMAC_Task3/Prediction of Spherical Equivalent/valid/mmac_task_3_val_0031.png", "/home/shuochen/MMAC_Task3/Prediction of Spherical Equivalent/valid/mmac_task_3_val_0123.png", "/home/shuochen/MMAC_Task3/Prediction of Spherical Equivalent/valid/mmac_task_3_val_0161.png",
                   "/home/shuochen/MMAC_Task3/Prediction of Spherical Equivalent/valid/mmac_task_3_val_0002.png", "/home/shuochen/MMAC_Task3/Prediction of Spherical Equivalent/valid/mmac_task_3_val_0103.png", "/home/shuochen/MMAC_Task3/Prediction of Spherical Equivalent/valid/mmac_task_3_val_0050.png",
                   "/home/shuochen/MMAC_Task3/Prediction of Spherical Equivalent/valid/mmac_task_3_val_0048.png", "/home/shuochen/MMAC_Task3/Prediction of Spherical Equivalent/valid/mmac_task_3_val_0200.png", "/home/shuochen/MMAC_Task3/Prediction of Spherical Equivalent/valid/mmac_task_3_val_0136.png"]

    for path in image_paths:
        if os.path.exists(path):
            img = cv2.imread(path)
            score = model.predict(img)
            print(f"Score for {path}: {score}")
        else:
            print(f"Image not found: {path}")