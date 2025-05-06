import os
import shutil
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from infer.bulk.infer import analyze_expression
import json
import numpy
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from flask import render_template

gene_visualize_bp = Blueprint('gene_visualize', __name__)

@gene_visualize_bp.route('/', methods=['GET'])
def index():
    return render_template('rna_visualize.html')


UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 存储 Plotly 数据
global_plotly_data = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@gene_visualize_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    return jsonify({'error': 'Invalid file type'}), 400

@gene_visualize_bp.route('/visualize', methods=['POST'])
def visualize_data():
    global global_plotly_data
    try:
        # 清空输出目录
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # 获取最新上传的文件
        uploaded_files = os.listdir(UPLOAD_FOLDER)
        if not uploaded_files:
            return jsonify({'error': 'No files available'}), 400
            
        latest_file = max(uploaded_files, key=lambda f: os.path.getctime(os.path.join(UPLOAD_FOLDER, f)))
        file_path = os.path.join(UPLOAD_FOLDER, latest_file)
        
        # 执行分析
        logger.debug(f"Analyzing file: {file_path}")
        global_plotly_data = analyze_expression(file_path, OUTPUT_FOLDER)
        logger.debug(f"Plotly data generated: {json.dumps(global_plotly_data, default=str)[:500]}...")
        
        return jsonify({
            'message': 'Visualization generated',
            'images': ['top_50_genes.png', 'gene_histogram.png', 'abnormal_genes_boxplot.png']  
        }), 200
        
    except Exception as e:
        logger.error(f"Error in visualize_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@gene_visualize_bp.route('/get_visualization_data', methods=['GET'])
def get_visualization_data():
    global global_plotly_data
    try:
        if global_plotly_data is None:
            logger.error("No Plotly data available")
            return jsonify({'error': 'No data available. Please generate visualizations first.'}), 400
        logger.debug(f"Returning Plotly data: {json.dumps(global_plotly_data, default=str)[:500]}...")
        return jsonify(global_plotly_data), 200
    except Exception as e:
        logger.error(f"Error in get_visualization_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@gene_visualize_bp.route('/output/<filename>')
def get_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)