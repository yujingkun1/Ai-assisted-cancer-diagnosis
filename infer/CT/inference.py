# inference.py
import os
import io
import base64
import numpy as np
from PIL import Image
from skimage import transform, filters
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry

# ------------------ 配置 ------------------
SAM_TYPE = "vit_b"
CKPT_PATH = "infer/MedSAM_main/work_dir/MedSAM/medsam_vit_b.pth"
INPUT_SIZE = 1024

# 设备
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_model = None
_embedding = None
_orig_img = None
_orig_h = None
_orig_w = None

# ------------------ 模型加载 ------------------
def load_model():
    global _model
    if _model is None:
        _model = sam_model_registry[SAM_TYPE](checkpoint=CKPT_PATH).to(DEVICE)
        _model.eval()
    return _model

# ------------------ 上传时：读取 & 预处理 ------------------
def read_image(file_stream, filename):
    """支持普通图片，也可扩展 DICOM/NIfTI"""
    img = Image.open(file_stream).convert("RGB")
    return np.array(img)

def encode_png_to_b64(np_img):
    buf = io.BytesIO()
    Image.fromarray(np_img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def get_image_embedding(np_img):
    """Resize->normalize->encode to get SAM image_embedding"""
    global _embedding, _orig_img, _orig_h, _orig_w
    _orig_img = np_img
    _orig_h, _orig_w = np_img.shape[:2]
    model = load_model()

    # resize & normalize
    img1024 = transform.resize(np_img, (INPUT_SIZE, INPUT_SIZE),
                               order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img1024 = (img1024 - img1024.min()) / max((img1024.max() - img1024.min()), 1e-8)
    tensor = (torch.from_numpy(img1024)
              .float()
              .permute(2, 0, 1)
              .unsqueeze(0)
              .to(DEVICE))
    with torch.no_grad():
        _embedding = model.image_encoder(tensor)
    return _orig_h, _orig_w, encode_png_to_b64(np_img)

# ------------------ 分割 ------------------
def segment_with_box(box):
    """
    box: [xmin, ymin, xmax, ymax] in 原始 H×W 坐标
    返回二值 mask np.uint8 (0/255)
    """
    if _embedding is None:
        raise ValueError("No image embedding available. Please upload an image first.")
    
    model = load_model()
    # 转到 1024 尺度
    box = np.array(box, dtype=float)
    box1024 = box / np.array([_orig_w, _orig_h, _orig_w, _orig_h]) * INPUT_SIZE
    box_t = torch.as_tensor(box1024[None, :], dtype=torch.float, device=DEVICE)

    with torch.no_grad():
        sparse, dense = model.prompt_encoder(points=None, boxes=box_t, masks=None)
        logits, _ = model.mask_decoder(
            image_embeddings=_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        prob = torch.sigmoid(logits)
        up = F.interpolate(prob,
                           size=(_orig_h, _orig_w),
                           mode="bilinear",
                           align_corners=False)
        mask = (up.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    return encode_png_to_b64(mask)

# ------------------ 预处理 ------------------
def preprocess_image(b64_png, method):
    data = base64.b64decode(b64_png)
    img = Image.open(io.BytesIO(data)).convert("L")
    arr = np.array(img)
    if method == "gaussian":
        out = filters.gaussian(arr, sigma=1)
    elif method == "median":
        out = filters.median(arr)
    elif method == "wavelet":
        # 简单示意：这里可以接 PyWavelets
        out = filters.gaussian(arr, sigma=0.5)  
    else:
        out = arr
    out = np.clip(out*255, 0, 255).astype(np.uint8)
    return encode_png_to_b64(out)


