import torch
import torchvision.models as models
from monai.bundle import ConfigParser
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, ToTensor

# 定义全局变量
device = None
unet_model = None
resnet_model = None
preprocess_transform = None

def init_models():
    global device, unet_model, resnet_model, preprocess_transform
    if device is None:  # 仅在第一次调用时初始化
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

        # 加载 spleen_ct_segmentation 模型
        try:
            # 使用 ConfigParser 加载模型配置
            config = ConfigParser()
            config.read_config("./model_weights/spleen_ct_segmentation/configs/inference.json")
            # 从配置文件中实例化网络
            unet_model = config.get_parsed_content("network")
            unet_model.to(device)
            # 加载预训练权重
            unet_model.load_state_dict(
                torch.load("./model_weights/spleen_ct_segmentation/models/model.pt", 
                           map_location=device, 
                           weights_only=True)
            )
            print("成功加载预训练 U-Net 模型: spleen_ct_segmentation")
        except Exception as e:
            print(f"加载 U-Net 模型失败: {str(e)}，使用默认模型")
            unet_model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            ).to(device)
        unet_model.eval()

        # ResNet 模型（使用 torchvision 的预训练权重）
        resnet_model = models.resnet18(weights='IMAGENET1K_V1')
        resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)  # 2 类：良性/恶性
        resnet_model.to(device)
        resnet_model.eval()

        # 图像预处理变换（用于 3D U-Net）
        preprocess_transform = Compose([
            LoadImage(image_only=True),      # 加载 3D 图像（支持 NIfTI 等格式）
            EnsureChannelFirst(),            # 确保通道优先
            ScaleIntensity(),                # 强度归一化
            ToTensor(),                      # 转换为张量
        ])
        print("模型和预处理变换初始化完成")

def check_models():
    print(f"检查模型状态 - unet_model: {unet_model}, device: {device}")