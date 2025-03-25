# blueprints/common.py
import torch
import torchvision.models as models
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, ToTensor

# 定义全局变量，但不立即加载模型
device = None
unet_model = None
resnet_model = None
preprocess_transform = None

def init_models():
    global device, unet_model, resnet_model, preprocess_transform
    if device is None:  # 仅在第一次调用时初始化
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # U-Net 模型（用于 CT 和 WSI 图像分割）
        unet_model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
        # unet_model.load_state_dict(torch.load("unet_ct_segmentation.pth", map_location=device))
        unet_model.eval()

        # ResNet 模型（用于分类）
        resnet_model = models.resnet18(weights='IMAGENET1K_V1')  # 修改 pretrained 为 weights
        resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)  # 2 类：良性/恶性
        # resnet_model.load_state_dict(torch.load("resnet_ct_classification.pth", map_location=device))
        resnet_model.to(device)
        resnet_model.eval()

        # 图像预处理变换（用于 U-Net）
        preprocess_transform = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            ToTensor(),
        ])