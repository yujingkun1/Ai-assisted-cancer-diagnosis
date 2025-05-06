from flask import Blueprint, request, jsonify, render_template
from flask_cors import CORS  # 导入 CORS 支持
import logging  # 导入日志模块
from infer.CT.inference import (
    read_image, get_image_embedding,
    segment_with_box, preprocess_image,
)

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

ct_image_bp = Blueprint("ct_image", __name__, url_prefix="/api/ct")
CORS(ct_image_bp)  # 启用 CORS 支持

@ct_image_bp.route('/', methods=['GET'])
def index():
    return render_template('ct_image.html')

@ct_image_bp.route("/upload", methods=["POST"])
def upload():
    """
    接收 form-data: key="file"
    返回 { height, width, image: base64_png }
    """
    try:
        file = request.files.get("file")
        if not file:
            logging.error("No file provided in request")
            return jsonify({"error": "No file provided"}), 400

        # 验证文件扩展名
        valid_extensions = {'.dcm', '.nii', '.nii.gz', '.png', '.jpg', '.jpeg'}
        if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
            logging.error(f"Unsupported file format: {file.filename}")
            return jsonify({"error": f"Unsupported file format. Supported: {valid_extensions}"}), 400

        logging.info(f"Processing file: {file.filename}")
        np_img = read_image(file.stream, file.filename)
        h, w, b64 = get_image_embedding(np_img)
        logging.info(f"Upload successful: height={h}, width={w}, image_len={len(b64)}")
        return jsonify({"height": h, "width": w, "image": b64})

    except Exception as e:
        logging.error(f"Upload failed: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@ct_image_bp.route("/segment", methods=["POST"])
def segment():
    """
    接收 JSON { box: [xmin,ymin,xmax,ymax] }
    返回 { mask: base64_png }
    """
    try:
        body = request.get_json()
        box = body.get("box")
        if not box or len(box) != 4:
            logging.error("Invalid or missing box parameter")
            return jsonify({"error": "Invalid box: must be [xmin,ymin,xmax,ymax]"}), 400

        logging.info(f"Segmenting with box: {box}")
        mask_b64 = segment_with_box(box)
        logging.info(f"Segmentation successful: mask_len={len(mask_b64)}")
        return jsonify({"mask": mask_b64})

    except Exception as e:
        logging.error(f"Segmentation failed: {str(e)}")
        return jsonify({"error": f"Segmentation failed: {str(e)}"}), 500

@ct_image_bp.route("/preprocess", methods=["POST"])
def preprocess():
    """
    接收 JSON { image: base64_png, method: str }
    返回 { image: base64_png }
    """
    try:
        body = request.get_json()
        img = body.get("image")
        method = body.get("method", "")
        if not img:
            logging.error("No image provided for preprocessing")
            return jsonify({"error": "No image provided"}), 400

        logging.info(f"Preprocessing with method: {method}")
        out_b64 = preprocess_image(img, method)
        logging.info(f"Preprocessing successful: image_len={len(out_b64)}")
        return jsonify({"image": out_b64})

    except Exception as e:
        logging.error(f"Preprocessing failed: {str(e)}")
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 500