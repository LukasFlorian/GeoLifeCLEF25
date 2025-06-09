import timm
import torch
import torch.nn as nn

class ResNet50(nn.Module):
    """
    Modified ResNet50 model for image classification.
    """
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("hf_hub:cm93/resnet50-eurosat", pretrained=True)
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2))
        self.model.fc = nn.Linear(self.model.fc.in_features, 11255, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet50 model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

class ResNet18(nn.Module):
    """
    ResNet18 model for image classification.
    """
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("hf_hub:cm93/resnet18-eurosat", pretrained=True)
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2))
        self.model.fc = nn.Linear(self.model.fc.in_features, 11255, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet18 model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)