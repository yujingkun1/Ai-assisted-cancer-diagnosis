from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os
from flask_cors import CORS

app = Flask(__name__)
# 显式配置 CORS，允许所有来源，或者指定前端来源
CORS(app, resources={r"/*": {"origins": "*"}})  # 允许所有来源
# 如果前端运行在特定地址（如 http://127.0.0.1:5500），可以指定：
# CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

@app.route('/')
def index():
    return app.send_static_file('table.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Received data: {data}")
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required_fields = ['age', 'gender_code', 'tumor.size', 'cea', 'grade_code', 'PNI', 'lymphcat']
        for field in required_fields:
            if field not in data:
                print(f"Missing field: {field}")
                return jsonify({'error': f'Missing field: {field}'}), 400

        age = data['age']
        gender_code = data['gender_code']
        tumor_size = data['tumor.size']
        cea = data['cea']
        grade_code = data['grade_code']
        PNI = data['PNI']
        lymphcat = data['lymphcat']

        try:
            age = float(age)
            gender_code = int(gender_code)
            tumor_size = float(tumor_size)
            cea = float(cea)
            grade_code = int(grade_code)
            PNI = int(PNI)
            lymphcat = int(lymphcat)
        except (ValueError, TypeError) as e:
            print(f"Data type error: {str(e)}")
            return jsonify({'error': 'Invalid data type for one or more fields', 'details': str(e)}), 400

        cmd = f"python infer/survival/infer.py {age} {gender_code} {tumor_size} {cea} {grade_code} {PNI} {lymphcat}"
        print(f"Executing command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Script failed with stderr: {result.stderr}")
            return jsonify({'error': 'Inference script failed', 'details': result.stderr}), 500

        output = result.stdout
        print(f"Script output: {output}")
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

        if risk_score is None or risk_group is None or survival_curve_path is None:
            return jsonify({'error': 'Failed to parse inference output', 'output': output}), 500

        survival_curve_relative_path = os.path.relpath(survival_curve_path, start=os.path.dirname(__file__))
        print(f"Survival curve path: {survival_curve_path}")

        return jsonify({
            'risk_score': risk_score,
            'risk_group': risk_group,
            'survival_curve_path': f"/output/survival/{os.path.basename(survival_curve_path)}"
        })

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/output/survival/<path:filename>')
def predictions(filename):
    try:
        return send_from_directory('output/survival', filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)