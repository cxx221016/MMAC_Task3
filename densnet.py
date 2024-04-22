import os
import cv2
import torch
from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import transforms

class DenseNet121(nn.Module):
    def __init__(self, out_size=1):
        super(DenseNet121, self).__init__()
        # DenseNet121 typically ends with 1024 features at the bottleneck layer
        feature_dim = 1024
        
        self.densenet121 = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        # Replacing classifier with a new one for our specific task
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(feature_dim, out_size)
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

class model:
    def __init__(self):
        self.checkpoint = "model_weights.pth"
        self.device = torch.device("cpu")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.386, 0.186, 0.024], [0.241, 0.125, 0.049])])
        
    def load(self, dir_path):
        self.model = DenseNet121(out_size=1)  # Use the new DenseNet model
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
    net = DenseNet121(out_size=1)
    x = torch.rand((32,3,512,512))
    y = net(x)
    print(y.shape)
