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
