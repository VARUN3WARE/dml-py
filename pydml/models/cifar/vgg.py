"""
VGG for CIFAR datasets.

Adapted from the VGG paper for CIFAR-10/100 (32x32 images).
Reference: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
Paper: https://arxiv.org/abs/1409.1556
"""

import torch
import torch.nn as nn


class VGG(nn.Module):
    """
    VGG network adapted for CIFAR datasets.
    
    Modified from the original VGG for 32x32 images instead of 224x224.
    Uses smaller feature maps and fewer pooling layers to maintain
    spatial resolution for small images.
    
    Args:
        vgg_name: VGG variant ('VGG11', 'VGG13', 'VGG16', 'VGG19')
        num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
        batch_norm: Whether to use batch normalization (default: True)
    """
    
    def __init__(self, vgg_name, num_classes=10, batch_norm=True):
        super(VGG, self).__init__()
        self.batch_norm = batch_norm
        self.features = self._make_layers(cfg[vgg_name])
        
        # Classifier adapted for CIFAR (32x32 -> after 5 pools: 1x1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.batch_norm:
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)
                    ]
                else:
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ]
                in_channels = x
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# VGG configurations
# Numbers represent output channels, 'M' represents max pooling
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(num_classes=10, batch_norm=True):
    """
    VGG-11 model for CIFAR datasets.
    
    Args:
        num_classes: Number of output classes (default: 10)
        batch_norm: Use batch normalization (default: True)
    
    Returns:
        VGG11 model instance
    
    Example:
        >>> from pydml.models.cifar import vgg11
        >>> model = vgg11(num_classes=100)  # For CIFAR-100
        >>> x = torch.randn(4, 3, 32, 32)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([4, 100])
    """
    return VGG('VGG11', num_classes=num_classes, batch_norm=batch_norm)


def vgg13(num_classes=10, batch_norm=True):
    """
    VGG-13 model for CIFAR datasets.
    
    Args:
        num_classes: Number of output classes (default: 10)
        batch_norm: Use batch normalization (default: True)
    
    Returns:
        VGG13 model instance
    
    Example:
        >>> from pydml.models.cifar import vgg13
        >>> model = vgg13(num_classes=100)  # For CIFAR-100
        >>> x = torch.randn(4, 3, 32, 32)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([4, 100])
    """
    return VGG('VGG13', num_classes=num_classes, batch_norm=batch_norm)


def vgg16(num_classes=10, batch_norm=True):
    """
    VGG-16 model for CIFAR datasets.
    
    Args:
        num_classes: Number of output classes (default: 10)
        batch_norm: Use batch normalization (default: True)
    
    Returns:
        VGG16 model instance
    
    Example:
        >>> from pydml.models.cifar import vgg16
        >>> model = vgg16(num_classes=10)  # For CIFAR-10
        >>> x = torch.randn(4, 3, 32, 32)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([4, 10])
    """
    return VGG('VGG16', num_classes=num_classes, batch_norm=batch_norm)


def vgg19(num_classes=10, batch_norm=True):
    """
    VGG-19 model for CIFAR datasets.
    
    Args:
        num_classes: Number of output classes (default: 10)
        batch_norm: Use batch normalization (default: True)
    
    Returns:
        VGG19 model instance
    
    Example:
        >>> from pydml.models.cifar import vgg19
        >>> model = vgg19(num_classes=100)  # For CIFAR-100
        >>> x = torch.randn(4, 3, 32, 32)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([4, 100])
    """
    return VGG('VGG19', num_classes=num_classes, batch_norm=batch_norm)
