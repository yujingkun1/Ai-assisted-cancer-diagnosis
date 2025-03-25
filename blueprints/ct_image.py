from flask import Blueprint, request, jsonify
import os
import base64
import cv2
import numpy as np
import pydicom
import pywt
import SimpleITK as sitk
from blueprints.common import unet_model, resnet_model, preprocess_transform, device, init_models

ct_image_bp = Blueprint('ct_image', __name__)

# 在蓝图初始化时调用 init_models
init_models()

@ct_image_bp.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        upload_dir = 'uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        if file.filename.endswith('.dcm'):
            dicom = pydicom.dcmread(file_path)
            image = dicom.pixel_array
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ct_image_bp.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        data = request.json
        image_base64 = data['image']
        method = data['method']

        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

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
            return jsonify({'error': 'Invalid preprocessing method'}), 400

        _, buffer = cv2.imencode('.png', processed_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ct_image_bp.route('/segment', methods=['POST'])
def segment():
    try:
        data = request.json
        image_base64 = data['image']

        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        image_tensor = preprocess_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            mask = unet_model(image_tensor)
            mask = torch.sigmoid(mask) > 0.5
            mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255

        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        colored_mask[mask == 255] = [0, 0, 255]
        segmented_image = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.7, colored_mask, 0.3, 0)

        _, buffer = cv2.imencode('.png', segmented_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ct_image_bp.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        image_base64 = data['image']

        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = resnet_model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

        classification = 'Malignant' if predicted_class == 1 else 'Benign'
        return jsonify({'classification': classification})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ct_image_bp.route('/reconstruct', methods=['POST'])
def reconstruct():
    try:
        data = request.json
        image_base64 = data['image']

        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        voxel_data = np.zeros((10, 10, 10), dtype=np.uint8)
        image_resized = cv2.resize(image, (10, 10))
        for z in range(10):
            voxel_data[:, :, z] = (image_resized > 128).astype(np.uint8) * 255

        voxel_data_list = voxel_data.tolist()
        return jsonify({'voxelData': voxel_data_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500