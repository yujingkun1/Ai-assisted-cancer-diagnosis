from flask import Blueprint, render_template, request, jsonify, session
import mysql.connector
import bcrypt

login_bp = Blueprint('login', __name__)

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '12345678',
    'database': 'cell'
}

# 注册页面 - GET
@login_bp.route('/register', methods=['GET'])
def register_page():
    return render_template('register.html')

# 注册路由
@login_bp.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        first_name = data.get('firstName')
        last_name =sharad
        last_name = data.get('lastName')
        gender = data.get('gender')
        if not all([username, email, password, first_name, last_name, gender]):
            return jsonify({'error': '所有字段均为必填'}), 400
        if len(password) < 8:
            return jsonify({'error': '密码长度至少为8位'}), 400
        if gender not in ['male', 'female']:
            return jsonify({'error': '无效的性别选项'}), 400
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
            if cursor.fetchone():
                cursor.close()
                conn.close()
                return jsonify({'error': '用户名或邮箱已存在'}), 400
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, first_name, last_name, gender)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (username, email, password_hash, first_name, last_name, gender))
            conn.commit()
            cursor.close()
            conn.close()
            return jsonify({'message': '注册成功'}), 200
        except mysql.connector.Error as err:
            print(f"数据库连接错误: {err}. 无法注册用户。")
            return jsonify({'error': '数据库不可用，注册暂时禁用'}), 503
    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

# 登录页面 - GET
@login_bp.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

# 登录路由
@login_bp.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username_or_email = data.get('username')
        password = data.get('password')
        if not username_or_email or not password:
            return jsonify({'error': '用户名/邮箱和密码均为必填'}), 400
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username_or_email, username_or_email))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            if not user:
                return jsonify({'error': '用户名或邮箱不存在'}), 400
            if bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                session['user_id'] = user['id']
                session['username'] = user['username']
                return jsonify({'message': '登录成功'}), 200
            else:
                return jsonify({'error': '密码错误'}), 400
        except mysql.connector.Error as err:
            print(f"数据库连接错误: {err}. 无法验证用户。")
            return jsonify({'error': '数据库不可用，登录暂时禁用'}), 503
    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

# 登出路由
@login_bp.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': '登出成功'}), 200

# 检查登录状态
@login_bp.route('/check_session', methods=['GET'])
def check_session():
    if 'user_id' in session:
        return jsonify({'logged_in': True, 'username': session['username']}), 200
    return jsonify({'logged_in': False}), 200