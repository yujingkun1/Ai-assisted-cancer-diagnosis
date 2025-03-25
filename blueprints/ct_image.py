from flask import Blueprint, request, jsonify
from flask_cors import CORS
import os
import base64
import cv2
import numpy as np
import nibabel as nib
import torch
from blueprints.common import unet_model, resnet_model, preprocess_transform, device, init_models
from monai.bundle import ConfigParser  # 确保导入 ConfigParser
from monai.networks.nets import UNet  # 确保导入 UNet
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, ToTensor

ct_image_bp = Blueprint('ct_image', __name__)

# 启用 CORS
CORS(ct_image_bp)

_unet_model = None

def get_unet_model():
    global _unet_model
    if _unet_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            config = ConfigParser()
            config.read_config("./model_weights/spleen_ct_segmentation/configs/inference.json")
            _unet_model = config.get_parsed_content("network")
            _unet_model.to(device)
            _unet_model.load_state_dict(
                torch.load("./model_weights/spleen_ct_segmentation/models/model.pt",
                           map_location=device, weights_only=True)
            )
            print("成功加载预训练 U-Net 模型: spleen_ct_segmentation")
        except Exception as e:
            print(f"加载失败: {e}，使用默认模型")
            _unet_model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            ).to(device)
        _unet_model.eval()
    return _unet_model


@ct_image_bp.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': '未上传文件'}), 400

        # 创建上传目录
        upload_dir = 'uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        filename_lower = file.filename.lower()
        if not filename_lower.endswith(('.nii', '.nii.gz')):
            return jsonify({'error': '仅支持 NIfTI 格式 (.nii 或 .nii.gz) 文件'}), 400

        # 读取 NIfTI 文件
        try:
            nii_image = nib.load(file_path)
            image = nii_image.get_fdata()  # 获取数据
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        except Exception as e:
            return jsonify({'error': f'读取 NIfTI 文件失败: {str(e)}'}), 500

        if image is None or len(image.shape) < 3:
            return jsonify({'error': '图像数据无效或维度不足'}), 500

        print(f"原始图像形状: {image.shape}")  # 调试输出

        # 处理不同维度的 NIfTI 文件
        if len(image.shape) == 3:  # [H, W, D]
            pass  # 已经是 3D 数据，无需处理
        elif len(image.shape) == 4:  # [H, W, D, C] 或 [C, H, W, D]
            # 判断通道维度位置
            if image.shape[3] < 5:  # 通道数较小，假设为 [H, W, D, C]
                image = image[:, :, :, 0]  # 取第一个通道
            else:  # 通道数较大，假设为 [C, H, W, D]
                image = image[0, :, :, :]  # 取第一个通道
        elif len(image.shape) == 5:  # [T, C, D, H, W] 或其他
            # 假设为 [T, C, D, H, W]，取第一个时间点和通道
            image = image[0, 0, :, :, :]
        else:
            return jsonify({'error': f'不支持的图像维度: {image.shape}'}), 500

        # 确保图像为 [H, W, D] 格式
        if image.shape[0] < image.shape[2]:  # 可能是 [D, H, W]
            image = np.transpose(image, (1, 2, 0))  # 转换为 [H, W, D]

        print(f"处理后图像形状: {image.shape}")  # 调试输出

        # 取中间切片用于显示
        mid_slice = image.shape[2] // 2
        image_slice = image[:, :, mid_slice]  # [H, W]
        print(f"image_slice shape: {image_slice.shape}, dtype: {image_slice.dtype}")  # 调试输出

        # 确保是单通道灰度图像
        image_slice = (image_slice * 255).astype(np.uint8)
        if len(image_slice.shape) != 2:
            return jsonify({'error': f'切片维度错误，期望 [H, W]，实际 {image_slice.shape}'}), 500

        # 使用 OpenCV 编码
        _, buffer = cv2.imencode('.png', image_slice)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': image_base64, 'full_image_path': file_path})

    except Exception as e:
        print(f"/upload 端点错误: {e}")
        return jsonify({'error': str(e)}), 500

@ct_image_bp.route('/segment', methods=['POST'])
def segment():
    try:
        data = request.get_json()
        full_image_path = data.get('full_image_path')
        if not full_image_path:
            return jsonify({'error': '未提供图像路径'}), 400

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

        # 加载模型（保持不变）
        try:
            from monai.bundle import ConfigParser
            config = ConfigParser()
            config.read_config("./model_weights/spleen_ct_segmentation/configs/inference.json")
            unet_model = config.get_parsed_content("network")
            unet_model.to(device)
            unet_model.load_state_dict(
                torch.load("./model_weights/spleen_ct_segmentation/models/model.pt",
                           map_location=device, weights_only=True)
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

        # 修正预处理流程（关键修改点）
        preprocess_transform = Compose([
            LoadImage(image_only=True),           # 原始形状 [H, W, D, C] 或 [H, W, D]
            EnsureChannelFirst(channel_dim=-1),   # 将最后一维转换为通道维度
            lambda x: x[:1] if x.shape[0] > 1 else x,  # 多通道情况取第一个通道
            ScaleIntensity(),
            lambda x: x.permute(0, 3, 1, 2) if x.ndim == 4 else x,  # 若为 4D 数据，调整为 [C, D, H, W]
            ToTensor(),
        ])

        print("开始执行分割...")
        if not os.path.exists(full_image_path):
            return jsonify({'error': f'文件路径 {full_image_path} 不存在'}), 400

        image_tensor = preprocess_transform(full_image_path)
        print(f"预处理后张量形状: {image_tensor.shape}")  # 例如 [1, D, H, W]

        image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 1, D, H, W]
        print(f"输入网络前的张量形状: {image_tensor.shape}")

        # 对深度方向进行 padding
        import torch.nn.functional as F
        original_depth = image_tensor.shape[2]
        padded_depth = int(np.ceil(original_depth / 16) * 16)
        pad_diff = padded_depth - original_depth
        pad_front = pad_diff // 2
        pad_back = pad_diff - pad_front
        image_tensor = F.pad(image_tensor, (0, 0, 0, 0, pad_front, pad_back), mode='constant', value=0)
        print(f"经过 padding 后张量形状: {image_tensor.shape}")

        with torch.no_grad():
            output = unet_model(image_tensor)
            print(f"模型输出形状: {output.shape}")  # [1, 2, padded_D, H, W]
            prediction = torch.argmax(output, dim=1)  # [1, padded_D, H, W]
            prediction = prediction.squeeze(0).cpu().numpy()  # [padded_D, H, W]

        prediction = prediction[pad_front: pad_front + original_depth, :, :]

        # 读取原始图像用于显示
        nii_image = nib.load(full_image_path)
        image = nii_image.get_fdata()
        if len(image.shape) == 4:
            image = image[:, :, :, 0]
        # 若图像为 [H, W, D]（例如 (320,320,20)），则转换为 [D, H, W]
        if image.shape[-1] < image.shape[0]:
            image = np.transpose(image, (2, 0, 1))
        
        # 取中间切片（以预测的深度为基准）
        mid_slice = prediction.shape[0] // 2
        mask_slice = prediction[mid_slice, :, :]
        image_slice = image[mid_slice, :, :]

        image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8) * 255
        image_slice = image_slice.astype(np.uint8)

        image_bgr = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2BGR)
        colored_mask = cv2.cvtColor(mask_slice.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        colored_mask[mask_slice > 0] = [0, 0, 255]

        if image_bgr.shape != colored_mask.shape:
            colored_mask = cv2.resize(colored_mask, (image_bgr.shape[1], image_bgr.shape[0]))
        
        segmented_image = cv2.addWeighted(image_bgr, 0.7, colored_mask, 0.3, 0)

        _, buffer = cv2.imencode('.png', segmented_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': image_base64})
    except Exception as e:
        print(f"/segment 端点错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# 保留 preprocess 和 classify 端点（根据需要调整）
@ct_image_bp.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        method = data.get('method')
        if not image_base64 or not method:
            return jsonify({'error': '缺少图像或预处理方法'}), 400

        # 将 Base64 转换为图像
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # 根据方法进行预处理
        if method == 'gaussian':
            processed_image = cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            processed_image = cv2.medianBlur(image, 5)
        elif method == 'wavelet':
            # 简单的小波变换示例（需要 pywt 库）
            import pywt
            coeffs = pywt.dwt2(image, 'haar')
            cA, (cH, cV, cD) = coeffs
            processed_image = pywt.idwt2((cA, (cH, cV, cD)), 'haar').astype(np.uint8)
        else:
            return jsonify({'error': '不支持的预处理方法'}), 400

        # 编码回 Base64
        _, buffer = cv2.imencode('.png', processed_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': image_base64})

    except Exception as e:
        print(f"/preprocess 端点错误: {e}")
        return jsonify({'error': str(e)}), 500

@ct_image_bp.route('/classify', methods=['POST'])
def classify():
    # 原有代码保持不变，或根据需要调整
    pass