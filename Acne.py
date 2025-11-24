# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  flask + YOLOv8 痘痘分析 API
#  (c) 2025 — 精簡至必要 import，移除 TensorFlow
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
import json
from flask import Flask, request, jsonify, send_from_directory
import cv2, numpy as np, os, pymysql, ssl
import requests
from werkzeug.utils import secure_filename
from ultralytics import YOLO               # ⬅️ 只用 Ultralytics
from datetime import datetime
from flask_cors import CORS

# ---------- 基本設定 ----------
app = Flask(__name__)                       # ❷ 拿掉 static_folder
CORS(app, resources={
    r"/*": {                              # 你的 API 現在沒有 /api 前綴，直接整站允許
        "origins": [
            "https://acnefrontend.onrender.com",
            "http://localhost:3000"
        ],
        "supports_credentials": False
    }
})
BASE_UPLOAD_FOLDER = 'uploads'
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = BASE_UPLOAD_FOLDER

DB_CONFIG = dict(
    host     = os.environ["DB_HOST"],
    port     = int(os.environ["DB_PORT"]),
    user     = os.environ["DB_USER"],
    password = os.environ["DB_PASS"],
    database = os.environ["DB_NAME"],
    charset  = "utf8mb4",
    ssl      = {"ca":  os.path.join(os.getcwd(), "ca.pem")}
)

# ---------- 載入 YOLO 模型 ----------
try:
    MODEL_PATH = os.path.join('detect', 'best.pt')     #  路徑是字串
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f'{MODEL_PATH} not found')
    model = YOLO(MODEL_PATH)                             #  建立 YOLO 物件
    print('✅ YOLO model loaded.')
except Exception as e:
    print(f'Error loading AI model: {e}')
    model = None

# 痘痘嚴重度對應表
acne_severity = {
    0: "Grade I: Mild acne with comedones.",
    1: "Grade II: Moderate acne with papules.",
    2: "Grade III: Severe acne with pustules.",
    3: "Grade IV: Very severe acne with nodules."
}

# ---------- 工具函式 ----------
def allowed_file(name): return '.' in name and name.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def classify_acne(img_path):
    """回傳分析結果：每類痘痘的數量與信心值"""
    if model is None:
        return 'AI model not loaded', 'N/A', {}

    try:
        res = model(img_path, conf=0.15, iou=0.4, verbose=False)[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return acne_severity[0], '0.00', 0, {}  # 沒抓到 → Grade I, conf=0, cnt=0, 空字典


        # 計算各類痘痘數量
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confs     = boxes.conf.cpu().numpy()
        class_map = {
            0: '粉刺 (Comedone)',
            1: '丘疹 (Papule)',
            2: '膿皰 (Pustule)',
            3: '結節 (Nodule)',
            4: '痘疤 (Scar)',
            5: '色素沉澱 (Dark Spot)' 
        }

        count_by_type = {}
        for cls_id, conf in zip(class_ids, confs):
            name = class_map.get(cls_id, f'類別 {cls_id}')
            if name not in count_by_type:
                count_by_type[name] = {'count': 0, 'max_conf': 0.0}
            count_by_type[name]['count'] += 1
            count_by_type[name]['max_conf'] = max(count_by_type[name]['max_conf'], conf)
        

        # 用總數分類嚴重度
        total = sum(v['count'] for v in count_by_type.values())
        if   total < 20: sev = 0
        elif total < 40: sev = 1
        elif total < 60: sev = 2
        else:            sev = 3
        cnt = total
        type_cnt = count_by_type
        for k, v in count_by_type.items():
            print(f"[Debug] 類別: {k}, 數量: {v['count']}, 信心值: {v['max_conf']:.2f}")
        
        return acne_severity[sev], f'{confs.max():.2f}', cnt, type_cnt
    except Exception as e:
        return f'Error during YOLO classification: {e}', 'N/A', {}

def save_user_folder(username):
    """確保本地資料夾 & user_folders 資料表存在"""
    try:
        path = os.path.join(BASE_UPLOAD_FOLDER, username)
        os.makedirs(path, exist_ok=True)
        with pymysql.connect(**DB_CONFIG) as conn:
            cur = conn.cursor()
            cur.execute('SELECT 1 FROM user_folders WHERE username=%s', (username,))
            if cur.fetchone() is None:
                cur.execute('INSERT INTO user_folders (username,folder_path) VALUES (%s,%s)', (username, path))
                conn.commit()
    except Exception as e:
        print('DB error(save_user_folder):', e)

def save_to_db(uid, fname, part, sev, conf, cnt, ts):
    try:
        conf_val = 0.0 if conf in [None, 'N/A'] else float(conf)
        with pymysql.connect(**DB_CONFIG) as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO acne_analysis
                  (user_id, filename, face_part, severity,
                   confidence, acne_count, upload_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (uid, fname, part, sev, conf_val, cnt, ts))
            conn.commit()
    except Exception as e:
        print('DB error(save_to_db):', e)

# ---------- 前端 SPA 入口 ----------
@app.route('/')
def index():
    return "Backend is running", 200
@app.route('/Chatbot')
@app.route('/Inform')
@app.route('/AnalysisResult')
def serve_react():
    return send_from_directory(app.static_folder, 'index.html')

# ---------- 上傳 API ----------
@app.route('/upload', methods=['POST'])
def upload():
    uid = secure_filename(request.form.get('user_id', 'anonymous'))
    save_user_folder(uid)
    folder = os.path.join(app.config['UPLOAD_FOLDER'], uid)
    ts = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S") #台灣地區時間
    results = []
    for part in ['left', 'middle', 'right']:
        f = request.files.get(part)
        if not f or not allowed_file(f.filename):
            return jsonify({'error': f'Invalid or missing file for {part}'}), 400

        fname = f'{uid}_{part}.jpg'
        path  = os.path.join(folder, fname)
        f.save(path)
        sev, conf, cnt ,type_cnt = classify_acne(path)  # ← 接收 dict 結果
        save_to_db(uid, fname, part, sev, conf, cnt, ts)  # 仍記錄總數

        # 將 type_cnt(有信心值) 內的 float32 轉換成標準 Python float
        for k, v in type_cnt.items():
            v['max_conf'] = float(v['max_conf'])
        
        # simple_type_cnt(沒信心值)
        simple_type_cnt = {k: {'count': v['count']} for k, v in type_cnt.items()}

        results.append({
            "face_part":  part,
            "filename":   fname,
            "severity":   sev,
            "confidence": conf,
            "acne_count": cnt,
            "acne_types": simple_type_cnt,
            "upload_time": ts
        })
    with open(os.path.join(folder, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print('✅ Upload & analysis done')           # 迴圈結束後總結
    return jsonify(success=True, user_id=uid, details=results)

# ---------- 其他輔助 API ----------
@app.route('/result')
def result():
    user_id = request.args.get('user_id')
    user_dir = os.path.join('uploads', user_id)
    result_path = os.path.join(user_dir, 'results.json')
    advice_path = os.path.join(user_dir, 'advice.html')

    if not os.path.exists(result_path):
        return jsonify({'error': 'results.json 不存在', 'results': [], 'advice': ''}), 404

    # 讀取結果
    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 若 advice.html 已存在，直接回傳
    if os.path.exists(advice_path):
        with open(advice_path, 'r', encoding='utf-8') as f:
            advice_html = f.read()
    else:
        # 組合 prompt 並呼叫 n8n 生成建議
        prompt = '\n'.join(f"{r['face_part']} face: {r['severity']}" for r in results)
        try:
            res = requests.post(
                os.environ["N8N_APP_API_URL"],
                json={"message": f"根據以下痘痘分析結果提供衛教建議和可參考資源：\n{prompt}"}
            )
            res.raise_for_status()
            reply_html = res.json().get('reply', '')

            # 儲存 HTML 格式建議
            with open(advice_path, 'w', encoding='utf-8') as f:
                f.write(reply_html)

            advice_html = reply_html
        except Exception as e:
            advice_html = f"<p>生成衛教建議失敗：{str(e)}</p>"

    return jsonify({
        'results': results,
        'advice': advice_html
    })

@app.route('/save-advice', methods=['POST'])
def save_advice():
    data = request.json
    user_id = data.get('user_id')
    advice = data.get('advice')

    folder = os.path.join('uploads', user_id)
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, 'advice.md'), 'w', encoding='utf-8') as f:
        f.write(advice)

    return jsonify({'success': True})


@app.route('/uploads/<user>/<filename>')
def uploaded_file(user, filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], user), filename)

@app.route('/check-user-id')
def check_user_id():
    uid = request.args.get('user_id')
    if not uid: return jsonify(error='Missing user_id'), 400
    with pymysql.connect(**DB_CONFIG) as conn:
        cur = conn.cursor()
        cur.execute('SELECT 1 FROM user_folders WHERE username=%s', (uid,))
        return jsonify(exists = cur.fetchone() is not None)

# ---------- 入口 ----------
if __name__ == "__main__":
    print('🚀 Flask server starting …')
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", 5000)),
        # debug=True               # 本地開 debug 方便追錯
    )



