from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os
from flask_cors import CORS  # 添加 CORS 支持

app = Flask(__name__)
CORS(app)  # 允许跨域请求

@app.route('/')
def index():
    """返回前端页面"""
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 接收前端发送的 JSON 数据
        data = request.json
        print(f"Received data: {data}")  # 调试：打印接收到的数据
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # 提取字段，确保所有必需字段都存在
        required_fields = ['age', 'gender_code', 'tumor_size', 'cea', 'grade_code', 'PNI', 'lymphcat']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        age = data['age']
        gender_code = data['gender_code']
        tumor_size = data['tumor_size']
        cea = data['cea']
        grade_code = data['grade_code']
        PNI = data['PNI']
        lymphcat = data['lymphcat']

        # 调用推理脚本 infer.py
        cmd = f"python survival_analysis/infer.py {age} {gender_code} {tumor_size} {cea} {grade_code} {PNI} {lymphcat}"
        print(f"Executing command: {cmd}")  # 调试：打印命令
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # 检查脚本是否成功执行
        if result.returncode != 0:
            print(f"Script failed with stderr: {result.stderr}")  # 调试：打印错误输出
            return jsonify({'error': 'Inference script failed', 'details': result.stderr}), 500

        # 解析脚本输出
        output = result.stdout
        print(f"Script output: {output}")  # 调试：打印脚本输出
        lines = output.split('\n')
        risk_score = None
        risk_group = None
        survival_curve_path = None

        for line in lines:
            if "患者风险分数" in line:
                risk_score = float(line.split(': ')[1])
            elif "风险分组" in line:
                risk_group = line.split(': ')[1]
            elif "生存曲线已保存为" in line:
                survival_curve_path = line.split(': ')[1].strip()

        # 确保所有必需结果都解析成功
        if risk_score is None or risk_group is None or survival_curve_path is None:
            return jsonify({'error': 'Failed to parse inference output', 'output': output}), 500

        # 规范化生存曲线路径为相对路径
        survival_curve_relative_path = os.path.relpath(survival_curve_path, start=os.path.dirname(__file__))
        print(f"Survival curve path: {survival_curve_path}")  # 调试：打印路径

        # 返回结果给前端
        return jsonify({
            'risk_score': risk_score,
            'risk_group': risk_group,
            'survival_curve_path': f"/predictions/{os.path.basename(survival_curve_path)}"
        })

    except Exception as e:
        print(f"Exception occurred: {str(e)}")  # 调试：打印异常
        return jsonify({'error': str(e)}), 500

@app.route('/predictions/<path:filename>')
def predictions(filename):
    try:
        return send_from_directory('survival_analysis/predictions', filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)