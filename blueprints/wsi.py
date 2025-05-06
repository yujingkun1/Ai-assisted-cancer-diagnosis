from flask import Blueprint, render_template, request, jsonify, send_from_directory, current_app
import os
import mysql.connector
from werkzeug.utils import secure_filename
import openslide
from PIL import Image
import io
import pandas as pd

wsi_bp = Blueprint('wsi', __name__)

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '12345678',
    'database': 'cell'
}

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'svs', 'ndpi', 'tif', 'tiff'}

# 检查文件扩展名
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert WSI to PNG
def convert_wsi_to_png(wsi_path, output_path, thumbnail_size=(1024, 1024)):
    try:
        slide = openslide.OpenSlide(wsi_path)
        thumbnail = slide.get_thumbnail(thumbnail_size)
        thumbnail.save(output_path, 'PNG')
        slide.close()
        return True
    except Exception as e:
        print(f"Error converting WSI to PNG: {e}")
        return False

# WSI 页面
@wsi_bp.route('/', methods=['GET'])
def wsi():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM gene_data")
        gene_data = cursor.fetchall()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"数据库连接错误: {err}. 返回空基因数据。")
        gene_data = []
    return render_template('WSI.html', gene_data=gene_data)

# 上传图片端点
@wsi_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        wsi_extensions = {'svs', 'ndpi', 'tif', 'tiff'}
        file_ext = filename.rsplit('.', 1)[1].lower()
        if file_ext in wsi_extensions:
            png_filename = filename.rsplit('.', 1)[0] + '.png'
            png_path = os.path.join(current_app.config['UPLOAD_FOLDER'], png_filename)
            if convert_wsi_to_png(file_path, png_path):
                os.remove(file_path)
                return jsonify({'message': 'WSI converted and uploaded successfully', 'filename': png_filename}), 200
            else:
                os.remove(file_path)
                return jsonify({'error': 'Failed to convert WSI to PNG'}), 500
        else:
            return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    return jsonify({'error': 'Invalid file type'}), 400

# 服务 uploads 目录中的文件
@wsi_bp.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

# 获取 DNA CSV
@wsi_bp.route('/get_dna_csv', methods=['GET'])
def get_dna_csv():
    csv_filename = 'dna.csv'
    csv_path = os.path.join(current_app.config['DATA_FOLDER'], csv_filename)
    if os.path.exists(csv_path):
        return send_file(csv_path, mimetype='text/csv')
    return jsonify({'error': 'dna.csv file not found'}), 404

# 获取 Mean Expression CSV
@wsi_bp.route('/get_mean_expression_csv', methods=['GET'])
def get_mean_expression_csv():
    csv_filename = 'mean_expression.csv'
    csv_path = os.path.join(current_app.config['DATA_FOLDER'], csv_filename)
    if os.path.exists(csv_path):
        return send_file(csv_path, mimetype='text/csv')
    return jsonify({'error': 'mean_expression.csv file not found'}), 404

# 非基因列
EXCLUDED_COLUMNS = [
    "unique_id", "image_name", "cell_id", "x", "y", "area", "perimeter", "cluster_label"
]

@wsi_bp.route('/get_top_genes', methods=['GET'])
def get_top_genes():
    csv_filename = 'pred_cells_4_25.csv'
    csv_path = os.path.join(current_app.config['DATA_FOLDER'], csv_filename)
    if not os.path.exists(csv_path):
        return jsonify({'error': 'pred_cells_4_25.csv file not found'}), 404
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        gene_columns = [col for col in df.columns if col not in EXCLUDED_COLUMNS]
        if not gene_columns:
            return jsonify({'error': 'No gene columns found in CSV after excluding specified columns'}), 400
        top_rows = df.head(10)
        gene_sums = top_rows[gene_columns].sum(numeric_only=True)
        gene_sums_dict = gene_sums.to_dict()
        gene_sum_list = [{'name': gene, 'value': float(value)} for gene, value in gene_sums_dict.items() if not pd.isna(value)]
        gene_sum_list.sort(key=lambda x: x['value'], reverse=True)
        top_genes = gene_sum_list[:10]
        print("Gene columns processed:", gene_columns)
        print("Top genes:", top_genes)
        return jsonify({'top_genes': top_genes}), 200
    except Exception as e:
        error_message = f'Failed to process pred_cells_4_25.csv: {str(e)}'
        print(error_message)
        return jsonify({'error': error_message}), 500