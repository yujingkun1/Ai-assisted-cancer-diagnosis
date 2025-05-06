from flask import Flask, render_template
from flask_cors import CORS
from flask_session import Session
import os
from blueprints.survival import survival_bp
from blueprints.ct_image import ct_image_bp
from blueprints.gene_visualize import gene_visualize_bp
from blueprints.wsi import wsi_bp
from blueprints.login import login_bp
from blueprints.common import init_models
import mysql.connector

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 配置文件夹
UPLOAD_FOLDER = 'Uploads'
DATA_FOLDER = 'data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER

# Session 配置
app.config['SECRET_KEY'] = 'e8b9d2f5a3c7b1e9f4d8c0a2b6e5f7d9c1a4b8e3f2d6c9a0b7e4f1d3c8a5b2e0'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '12345678',
    'database': 'cell'
}

# 数据库初始化 - 创建用户表
def init_db():
    try:
        conn = mysql.connector.connect(**db_config)
        print("Database connected successfully")
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                first_name VARCHAR(50) NOT NULL,
                last_name VARCHAR(50) NOT NULL,
                gender ENUM('male', 'female') NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gene_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255)
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"数据库初始化错误: {err}. 跳过数据库初始化，继续运行。")
        # 不抛出异常，继续运行

# 主路由
@app.route('/')
def index():
    return render_template('index.html')

# 注册蓝图
app.register_blueprint(survival_bp, url_prefix='/survival')
app.register_blueprint(ct_image_bp, url_prefix='/api/ct')
app.register_blueprint(gene_visualize_bp, url_prefix='/rna_visualize')
app.register_blueprint(wsi_bp, url_prefix='/wsi')
app.register_blueprint(login_bp, url_prefix='/auth')

if __name__ == '__main__':
    init_db()
    print("初始化模型...")
    init_models()
    print("模型初始化完成")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)