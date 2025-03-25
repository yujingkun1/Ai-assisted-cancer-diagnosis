from flask import Blueprint, request, jsonify
from flask_cors import CORS
import os
import base64
import cv2
import numpy as np
import pydicom
import pywt
import SimpleITK as sitk
import nibabel as nib
import torch
from blueprints.common import unet_model, resnet_model, preprocess_transform, device, init_models

ct_image_bp = Blueprint('ct_image', __name__)

# 启用 CORS
CORS(ct_image_bp)

# 在蓝图初始化时调用 init_models
try:
    init_models()
    # 不需要检查 unet_model 或 resnet_model 是否为 None，因为 init_models 总是会初始化它们
    print("模型初始化成功")
except Exception as e:
    print(f"模型初始化错误: {e}")
    # 如果初始化失败，后续请求会明确返回模型未初始化的错误

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
        image = None

        # 处理 DICOM 格式 (.dcm)
        if filename_lower.endswith('.dcm'):
            try:
                dicom = pydicom.dcmread(file_path)
                image = dicom.pixel_array
                if len(image.shape) > 2:  # 多帧 DICOM
                    image = image[0]  # 取第一帧
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            except Exception as e:
                return jsonify({'error': f'读取 DICOM 文件失败: {str(e)}'}), 500

        # 处理 NIfTI 格式 (.nii 或 .nii.gz)
        elif filename_lower.endswith(('.nii', '.nii.gz')):
            try:
                nii_image = nib.load(file_path)
                image = nii_image.get_fdata()
                if len(image.shape) == 3:  # 3D 数据
                    mid_slice = image.shape[2] // 2
                    image = image[:, :, mid_slice]
                elif len(image.shape) > 3:  # 4D 数据
                    mid_slice = image.shape[2] // 2
                    image = image[:, :, mid_slice, 0]
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            except Exception as e:
                return jsonify({'error': f'读取 NIfTI 文件失败: {str(e)}'}), 500

        # 处理 MHD 格式 (.mhd)
        elif filename_lower.endswith('.mhd'):
            try:
                mhd_image = sitk.ReadImage(file_path)
                image = sitk.GetArrayFromImage(mhd_image)
                if len(image.shape) == 3:  # 3D 数据
                    mid_slice = image.shape[0] // 2
                    image = image[mid_slice, :, :]
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            except Exception as e:
                return jsonify({'error': f'读取 MHD 文件失败: {str(e)}'}), 500

        # 处理普通图像格式 (.png, .jpg, .jpeg, .bmp, .tiff)
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    return jsonify({'error': '无法读取图像文件'}), 500
            except Exception as e:
                return jsonify({'error': f'读取图像文件失败: {str(e)}'}), 500

        else:
            return jsonify({'error': f'不支持的文件格式: {file.filename}'}), 400

        if image is None:
            return jsonify({'error': '图像处理失败'}), 500

        # 转换为 Base64
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': image_base64})

    except Exception as e:
        print(f"/upload 端点错误: {e}")
        return jsonify({'error': str(e)}), 500

@ct_image_bp.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        method = data.get('method')
        if not image_base64 or not method:
            return jsonify({'error': '缺少图像数据或预处理方法'}), 400

        # 解码图像
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({'error': '图像解码失败'}), 500

        # 预处理
        if method == 'gaussian':
            processed_image = cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            processed_image = cv2.medianBlur(image, 5)
        elif method == 'wavelet':
            coeffs = pywt.dwt2(image, 'haar')
            cA, (cH, cV, cD) = coeffs
            processed_image = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
            processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
        else:
            return jsonify({'error': '无效的预处理方法'}), 400

        _, buffer = cv2.imencode('.png', processed_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': image_base64})

    except Exception as e:
        print(f"/preprocess 端点错误: {e}")
        return jsonify({'error': str(e)}), 500

@ct_image_bp.route('/segment', methods=['POST'])
def segment():
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        if not image_base64:
            return jsonify({'error': '未提供图像数据'}), 400

        # 解码图像
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({'error': '图像解码失败'}), 500

        # 检查模型
        if unet_model is None:
            return jsonify({'error': 'U-Net 模型未初始化'}), 500

        # 保存临时文件以适配 MONAI 的 LoadImage
        temp_path = 'temp_image.png'
        cv2.imwrite(temp_path, image)
        image_tensor = preprocess_transform(temp_path).unsqueeze(0).to(device)
        os.remove(temp_path)  # 删除临时文件

        with torch.no_grad():
            mask = unet_model(image_tensor)
            mask = torch.sigmoid(mask) > 0.5
            mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255

        # 生成分割图像
        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        colored_mask[mask == 255] = [0, 0, 255]  # 红色
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        segmented_image = cv2.addWeighted(image_bgr, 0.7, colored_mask, 0.3, 0)

        _, buffer = cv2.imencode('.png', segmented_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': image_base64})

    except Exception as e:
        print(f"/segment 端点错误: {e}")
        return jsonify({'error': str(e)}), 500

@ct_image_bp.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        if not image_base64:
            return jsonify({'error': '未提供图像数据'}), 400

        # 解码图像
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({'error': '图像解码失败'}), 500

        # 检查模型
        if resnet_model is None:
            return jsonify({'error': 'ResNet 模型未初始化'}), 500

        # 图像预处理
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image / 255.0  # 归一化
        image = np.transpose(image, (2, 0, 1))  # 调整为 (C, H, W)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

        # 模型推理
        with torch.no_grad():
            logits = resnet_model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

        return jsonify({'class': predicted_class, 'probability': probs[0][predicted_class].item()})

    except Exception as e:
        print(f"/classify 端点错误: {e}")
        return jsonify({'error': str(e)}), 500