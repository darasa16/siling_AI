# ====================================================
# === BAGIAN 1: IMPORT LIBRARY & SETUP GLOBAL/KONSTANTA ===
# ====================================================

import RPi.GPIO as GPIO
import time
import spidev
import board
import adafruit_dht
from adafruit_dht import DHT22
import math
from flask import Flask, render_template, jsonify, request
import socket, requests, psutil, subprocess
import os
import uuid
import torch
from ultralytics import YOLO 
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
import os

# optional: OpenCV check (agar error import terlihat saat startup, bukan saat request)
try:
    import cv2
    HAVE_OPENCV = True
except Exception:
    HAVE_OPENCV = False

# Pastikan folder hasil ada (YOLO menyimpan 'static/results/detections')
os.makedirs("static/results/detections", exist_ok=True)

# --- Setup GPIO ---
try:
    GPIO.setmode(GPIO.BCM)
    print("✅ Mode GPIO ditetapkan ke BCM.")
except RuntimeError as e:
    if "already been set" in str(e):
        print("⚠ Mode GPIO sudah diatur sebelumnya. Melanjutkan.")
    else:
        raise e

# --- Digital Pin Sensor Gas ---
MQ2_DO = 5
MQ135_DO = 27
GPIO.setup(MQ2_DO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(MQ135_DO, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# --- DHT22 ---
DHT_PIN = board.D17
dhtDevice = adafruit_dht.DHT22(DHT_PIN, use_pulseio=False)

# --- Setup MCP3008 ---
spi = spidev.SpiDev()
spi.open(0, 0) # Bus=0, CE0
spi.max_speed_hz = 1350000

# --- Konstanta Kalibrasi Sensor ---
V_REF = 3.3
R_LOAD = 10.0 # kΩ

# A. Kalibrasi pH
V_PH7 = 1.681
V_PH4 = 2.10
SLOPE = (4.0 - 7.0) / (V_PH4 - V_PH7)
OFFSET = 7.0 - SLOPE * V_PH7

# B. Konstanta MQ SENSOR
A_MQ2 = 980.0
B_MQ2 = -1.78
MQ2_RATIO_CLEAN_AIR = 9.83

A_MQ135 = 110.0
B_MQ135 = -2.60
MQ135_RATIO_CLEAN_AIR = 3.7

# C. Assign channel MCP3008
CH_MQ2_AO = 0
CH_MQ135_AO = 1
CH_LDR = 2
CH_PH = 3

# --- Setup AI (YOLOv8) ---
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_PATH = "runs/train/deteksi_jentik_part7/weights/best.pt"

print("Memuat model YOLOv8...")
device = 0 if torch.cuda.is_available() else "cpu"
try:
    model = YOLO(MODEL_PATH)
    print(f"Model berhasil dimuat di device: {device}")
except Exception as e:
    print(f"Gagal memuat model AI: {e}. Fungsionalitas AI mungkin tidak tersedia.")
    model = None 

# --- Variabel Global untuk Metrik & Kalibrasi ---
app = Flask(__name__)
# ⬇️ MEMBACA FILE .env ⬇️
load_dotenv()
API_KEY = os.getenv("API_KEY")

# ====================================================
# === AUTENTIKASI API KEY (Keamanan Backend) ===
# ====================================================
from functools import wraps

def require_api_key(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = request.headers.get("X-API-KEY")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized access"}), 401
        return func(*args, **kwargs)
    return wrapper

R0_MQ2 = 25.0
R0_MQ135 = 25.0
request_count = 0
start_time = time.time()
try:
    net_counters = psutil.net_io_counters()
    last_bytes_sent = net_counters.bytes_sent
    last_bytes_recv = net_counters.bytes_recv
    last_time = time.time()
except Exception:
    last_bytes_sent, last_bytes_recv, last_time = 0, 0, time.time()


# ====================================================
# === BAGIAN 2: FUNGSI UTILITY SENSOR ===
# ====================================================

def read_adc(channel):
    """Membaca nilai ADC dari MCP3008."""
    if channel < 0 or channel > 7:
        return -1
    r = spi.xfer2([1, (8+channel)<<4, 0])
    adc_out = ((r[1] & 3) << 8) + r[2]
    return adc_out

def read_resistance(adc_val, V_ref=V_REF, R_load=R_LOAD):
    """Menghitung Rs (Resistansi Sensor) dari nilai ADC."""
    V_out = (adc_val * V_ref) / 1023.0
    if V_out <= 0:
        return float('inf')
    Rs = ((V_ref / V_out) - 1.0) * R_load
    return Rs

def resistance_to_ppm_mq(Rs, R0, A, B):
    """Konversi Rs ke ppm menggunakan Model Daya: ppm = A * (Rs/R0)^B."""
    try:
        ratio = Rs / R0 if R0 and R0 > 0 else 0
        if ratio <= 0:
            return None
        ppm = A * (ratio ** B)
        return round(ppm, 2)
    except Exception:
        return None

def resistance_to_ppm_mq2(Rs, R0):
    return resistance_to_ppm_mq(Rs, R0, A_MQ2, B_MQ2)

def resistance_to_ppm_mq135(Rs, R0):
    return resistance_to_ppm_mq(Rs, R0, A_MQ135, B_MQ135)

def adc_to_lux(adc_val):
    """Konversi LDR (dengan pembalikan logika)."""
    if adc_val < 0:
        return 0.0
    inverted_adc = 1023.0 - adc_val
    return max(0.0, (inverted_adc / 1023.0) * 1000) # Skala 0-1000 Lux

def calibrate_sensor(channel, ratio_clean_air, samples=50, delay=0.2):
    """Menghitung R0 (Resistansi di Udara Bersih)."""
    Rs_sum = 0.0
    for i in range(samples):
        adc_val = read_adc(channel)
        Rs = read_resistance(adc_val)
        Rs_sum += Rs
        time.sleep(delay)
    Rs_avg = Rs_sum / samples
    return Rs_avg / ratio_clean_air

def setup_calibration():
    """Fungsi untuk menjalankan kalibrasi R0 saat startup."""
    global R0_MQ2, R0_MQ135
    print("=== Kalibrasi MQ-2 & MQ-135 saat startup... ===")
    try:
        R0_MQ2 = calibrate_sensor(CH_MQ2_AO, MQ2_RATIO_CLEAN_AIR)
        R0_MQ135 = calibrate_sensor(CH_MQ135_AO, MQ135_RATIO_CLEAN_AIR)
        print(f"R0 MQ-2: {R0_MQ2:.2f} kΩ, R0 MQ-135: {R0_MQ135:.2f} kΩ. Selesai.")
    except Exception as e:
        print(f"Gagal Kalibrasi R0: {e}. Menggunakan nilai default.")


# ====================================================
# === BAGIAN 3: FUNGSI JARINGAN & SENSOR UTAMA ===
# ====================================================

def network_analysis():
    """Mengumpulkan metrik jaringan."""
    global last_bytes_sent, last_bytes_recv, last_time

    hostname = socket.gethostname()
    ip_local = socket.gethostbyname(hostname)

    try:
        ip_public = requests.get("https://api.ipify.org", timeout=1).text
    except:
        ip_public = "Tidak bisa ambil"

    interfaces = psutil.net_if_stats()
    net_io = psutil.net_io_counters()
    bytes_sent = net_io.bytes_sent
    bytes_recv = net_io.bytes_recv

    now = time.time()
    elapsed = now - last_time
    if elapsed > 0.01:
        tx_speed = (bytes_sent - last_bytes_sent) / elapsed / 1024
        rx_speed = (bytes_recv - last_bytes_recv) / elapsed / 1024
    else:
        tx_speed, rx_speed = 0.0, 0.0

    last_bytes_sent, last_bytes_recv, last_time = bytes_sent, bytes_recv, now

    try:
        ping = subprocess.check_output(
            ["ping", "-c", "1", "-W", "1", "8.8.8.8"],
            universal_newlines=True
        )
        latency_line = [line for line in ping.split('\n') if 'avg' in line]
        if latency_line:
            latency = latency_line[0].split('/')[4] + " ms"
        else:
            latency = "Timeout"
    except:
        latency = "Timeout"

    return {
        "ip_local": ip_local,
        "ip_public": ip_public,
        "interfaces": {name: stats.isup for name, stats in interfaces.items()},
        "tx_total": f"{bytes_sent/1024/1024:.2f} MB",
        "rx_total": f"{bytes_recv/1024/1024:.2f} MB",
        "tx_speed": f"{tx_speed:.2f} KB/s",
        "rx_speed": f"{rx_speed:.2f} KB/s",
        "latency": latency
    }

def get_all_sensor_readings():
    """Membaca semua sensor dan mengembalikan data dalam dictionary."""

    # --- DHT22 (Suhu & Kelembaban) ---
    temperature_c, humidity = None, None
    try:
        temperature_c = dhtDevice.temperature
        humidity = dhtDevice.humidity
    except RuntimeError:
        pass 
    except Exception:
        pass

    # --- MQ2 (Gas Mudah Terbakar) ---
    mq2_status = "🚨 Gas terdeteksi" if GPIO.input(MQ2_DO) == GPIO.LOW else "✅ Aman"
    mq2_adc = read_adc(CH_MQ2_AO)
    Rs_mq2 = read_resistance(mq2_adc)
    mq2_ppm = resistance_to_ppm_mq2(Rs_mq2, R0_MQ2)

    # --- MQ135 (Kualitas Udara) ---
    mq135_status = "⚠ Udara buruk" if GPIO.input(MQ135_DO) == GPIO.LOW else "✅ Udara baik"
    mq135_adc = read_adc(CH_MQ135_AO)
    Rs_mq135 = read_resistance(mq135_adc)
    mq135_ppm = resistance_to_ppm_mq135(Rs_mq135, R0_MQ135)

    # --- LDR (Cahaya) ---
    ldr_adc = read_adc(CH_LDR)
    ldr_status = "🌞 Terang" if ldr_adc < 300 else "🌑 Gelap"
    ldr_lux = adc_to_lux(ldr_adc)

    # --- pH ---
    ph_adc = read_adc(CH_PH)
    ph_v = (ph_adc * V_REF) / 1023.0
    ph_val = SLOPE * ph_v + OFFSET

    # --- NETWORK ---
    net_info = network_analysis()

    return {
        # Data untuk tampilan Web (dengan format string)
        "temp": f"{temperature_c:.1f}" if temperature_c is not None else "Error",
        "hum": f"{humidity:.1f}" if humidity is not None else "Error",
        "mq2_status": mq2_status,
        "mq2_ppm": f"{mq2_ppm}" if mq2_ppm is not None else "Error",
        "mq135_status": mq135_status,
        "mq135_ppm": f"{mq135_ppm}" if mq135_ppm is not None else "Error",
        "ldr_status": ldr_status,
        "ldr_lux": f"{ldr_lux:.2f}",
        "ldr_adc": f"{ldr_adc}", # Ini sudah benar, dipertahankan
        "ph_val": f"{ph_val:.2f}",
        
        # Data MENTAH untuk penyimpanan DB (nilai float/None)
        "temp_raw": temperature_c,
        "hum_raw": humidity,
        "ph_val_raw": ph_val,
        "ldr_lux_raw": ldr_lux,
        "mq2_ppm_raw": mq2_ppm,
        "mq135_ppm_raw": mq135_ppm,
        # ⬇️ PERBAIKAN: Menambahkan ldr_adc_raw untuk DB ⬇️
        "ldr_adc_raw": ldr_adc, 
        
        "net": net_info
    }


# ====================================================
# === BAGIAN 4: FLASK ENDPOINTS (RUTE WEB) ===
# ====================================================

@app.route("/heartbeat")
def heartbeat():
    """Endpoint sederhana untuk pengecekan server/alive."""
    return jsonify({"status": "ok"})
    
@app.route("/")
def index():
    """Rute utama untuk menampilkan data sensor (Halaman Web)."""
    global request_count, start_time
    request_count += 1
    elapsed = time.time() - start_time
    throughput = request_count / elapsed if elapsed > 0 else 0

    data = get_all_sensor_readings()
    net_info = data["net"]

    # === LOG TERMINAL ===
    print("=" * 40)
    print(f"Request: {request_count}, Throughput: {throughput:.2f} req/s")
    print(f"DHT22: {data['temp']}, {data['hum']}")
    print(f"MQ2: {data['mq2_status']} | {data['mq2_ppm']}")
    print(f"MQ135: {data['mq135_status']} | {data['mq135_ppm']}")
    print(f"LDR: {data['ldr_status']} | ADC: {data['ldr_adc']} | Lux: {data['ldr_lux']}")
    print(f"pH: {data['ph_val']}")
    print(f"IP Lokal: {net_info['ip_local']}, Latency: {net_info['latency']}")
    print("=" * 40)

    data_for_template = {k: v for k, v in data.items() if not k.endswith('_raw')}
    
    return render_template(
        "index.html",
        **data_for_template,
        throughput=f"{throughput:.2f} req/s"
    )


@app.route("/data")
@require_api_key
def get_data_json():
    """Endpoint untuk AJAX, mengembalikan data sensor mentah dalam JSON."""
    data = get_all_sensor_readings()
    
    # Simpan data LENGKAP ke database
    save_sensor_data(data)
    
    # Hapus data mentah (_raw) sebelum dikirim ke frontend
    data_for_json = {k: v for k, v in data.items() if not k.endswith('_raw')}
    
    return jsonify(data_for_json)


@app.route("/trigger_ai", methods=["POST"])
@require_api_key
def trigger_ai():
    """Endpoint untuk mengambil gambar, jalankan YOLO,
        simpan hasil ke DB Supabase, dan kembalikan JSON.
    """
    if model is None:
        return jsonify({"error": "Model AI tidak tersedia di server."}), 500

    # 1) Persiapan file
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # 2) Ambil gambar (subprocess)
    try:
        from shutil import which
        if which("rpicam-jpeg") is None and which("rpicam-still") is None:
            return jsonify({"error": "Perintah rpicam tidak ditemukan (install rpicam-apps)."}), 500

        cmd = [
            "rpicam-jpeg", "-o", filepath, "-t", "2000",
            "--width", "1280", "--height", "720",
            "--sharpness", "15",
            "--awb", "auto"
        ]
        if which("rpicam-jpeg") is None:
            cmd = [
                "rpicam-still", "-o", filepath, "-t", "2000",
                "--width", "1280", "--height", "720"
            ]
        proc = subprocess.run(cmd, timeout=8, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"rpicam gagal: {proc.stderr.strip()}")
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Timeout saat mengambil gambar dari kamera."}), 500
    except Exception as e:
        print(f"[CAMERA ERROR] {e}")
        return jsonify({"error": f"Gagal ambil gambar: {e}"}), 500

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        print("[CAMERA] File gambar tidak dibuat atau kosong.")
        return jsonify({"error": "Gambar kamera tidak tersedia."}), 500


    # 3) Pre-process gambar (sharpen/contrast)
    try:
        if HAVE_OPENCV:
            img = cv2.imread(filepath)
            if img is None:
                raise RuntimeError("cv2 gagal membaca file gambar.")
            blur = cv2.GaussianBlur(img, (0, 0), 3)
            img_sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
            alpha, beta = 1.2, 10
            img_enh = cv2.convertScaleAbs(img_sharp, alpha=alpha, beta=beta)
            cv2.imwrite(filepath, img_enh)
            print("[IMAGE] preprocessing (opencv) sukses.")
        else:
            from PIL import Image, ImageEnhance
            img = Image.open(filepath)
            img = ImageEnhance.Contrast(img).enhance(1.5)
            img = ImageEnhance.Sharpness(img).enhance(1.3)
            img = ImageEnhance.Brightness(img).enhance(1.05)
            img.save(filepath)
            print("[IMAGE] preprocessing (Pillow) sukses.")
    except Exception as e:
        print(f"[IMAGE WARN] preprocessing gagal: {e}")

    # 4) Jalankan deteksi YOLO
    try:
        results = model.predict(
            source=filepath, save=True, project="static/results",
            name="detections", exist_ok=True, device=device,
            conf=0.20, iou=0.45, verbose=False
        )
    except Exception as e:
        print(f"[AI ERROR] Gagal inferensi YOLO: {e}")
        return jsonify({"error": f"Gagal menjalankan model AI: {e}"}), 500

    # 5) Baca hasil inference
    num = 0
    try:
        if len(results) > 0 and hasattr(results[0], "boxes"):
            num = len(results[0].boxes) if results[0].boxes is not None else 0
        print("[AI] Jumlah deteksi:", num)
    except Exception as e:
        print(f"[AI WARN] Gagal baca box hasil: {e}")

    # 6) Path hasil image
    result_image_path = os.path.join("static", "results", "detections", os.path.basename(filepath))
    if not os.path.exists(result_image_path):
        result_image_path = filepath
    
    status_text = "Jentik Terdeteksi" if num > 0 else "Tidak Terdeteksi"
    status_response = "✅ Ada jentik terdeteksi." if num > 0 else "❌ Tidak ada jentik terdeteksi."

    # ==========================================================
    # === ⬇️ PERBAIKAN: Simpan hasil AI ke iotuser.ai_runs ⬇️ ===
    # ==========================================================
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            image_path_for_db = f"/{result_image_path}"
            
            cur.execute("""
                INSERT INTO iotuser.ai_runs 
                (device_id, "timestamp", total_detections, status, result_image)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                "RPI-PBL-01", 
                datetime.now(), 
                num, 
                status_text, 
                image_path_for_db
            ))
            conn.commit()
            cur.close()
            conn.close()
            print(f"✅ Data AI (Status: {status_text}, Num: {num}) berhasil disimpan ke Supabase.")
        except Exception as e:
            print(f"❌ Error simpan hasil AI ke Supabase: {e}")
            if conn:
                conn.rollback()
    else:
        print("⚠ Tidak bisa konek ke Supabase, hasil AI tidak disimpan.")
    # ==========================================================
    # === ⬆️ AKHIR PERBAIKAN ⬆️ ===
    # ==========================================================
    
    # 8) Kembalikan hasil ke frontend
    return jsonify({
        "status": status_response,
        "num_detections": num,
        "result_image": f"/{result_image_path}" # Path ini yang akan dibaca frontend
    })


# ====================================================
# === BAGIAN TAMBAHAN: KONEKSI SUPABASE ===
# ====================================================

# ⬇️ PERBAIKAN: Ambil variabel env untuk koneksi ⬇️
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT", "5432") # Default 5432 jika tidak ada

def get_db_connection():
    """Mendapatkan koneksi ke database Supabase PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        return conn
    except Exception as e:
        print(f"❌ Gagal konek ke Supabase DB: {e}")
        return None

def get_data_value(data, key):
    """
    Fungsi utilitas untuk mengambil nilai float dari data mentah/raw.
    Mengembalikan float atau None jika data tidak valid (NULL, Error, atau format salah).
    """
    raw_key = key + "_raw"
    val = data.get(raw_key)
    
    if val is None:
        return None 
    
    try:
        if isinstance(val, str) and val.lower() == 'error':
            return None
        return float(val)
    except (TypeError, ValueError):
        return None 

def save_sensor_data(data):
    """
    Simpan data sensor ke tabel iotuser.sensor_readings di Supabase.
    """
    conn = get_db_connection()
    if not conn:
        print("⚠ Tidak bisa konek ke Supabase, data sensor tidak disimpan.")
        return

    try:
        # ⬇️ PERBAIKAN: Ambil data _raw untuk skema iotuser.sensor_readings ⬇️
        temp_c = get_data_value(data, 'temp')
        humidity = get_data_value(data, 'hum')
        mq2_ppm = get_data_value(data, 'mq2_ppm')
        mq135_ppm = get_data_value(data, 'mq135_ppm')
        ph_val = get_data_value(data, 'ph_val')
        ldr_adc = get_data_value(data, 'ldr_adc') # Menggunakan ldr_adc_raw yang baru
        
        cur = conn.cursor()
        
        # ⬇️ PERBAIKAN: SQL INSERT untuk tabel iotuser.sensor_readings ⬇️
        insert_query = """
        INSERT INTO iotuser.sensor_readings 
        (device_id, temp_c, humidity, mq2_ppm, mq135_ppm, ph_val, ldr_adc, "timestamp")
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cur.execute(insert_query, (
            "RPI-PBL-01", # ID Perangkat Keras
            temp_c, 
            humidity, 
            mq2_ppm, 
            mq135_ppm,
            ph_val,
            ldr_adc,
            datetime.now() # Timestamp saat ini
        ))
        
        conn.commit()
        cur.close()
        print(f"✅ Data sensor berhasil disimpan ke Supabase (iotuser.sensor_readings).")
        
    except psycopg2.Error as e:
        print(f"❌ Error simpan data sensor (DB): {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"❌ Error simpan data sensor (Logika Python): {e}")
    finally:
        if conn:
            conn.close()

# ⬇️ FUNGSI INI (save_ai_data) TIDAK DIGUNAKAN LAGI, logikanya dipindah ke /trigger_ai
# def save_ai_data(label, confidence):
#    ...

# ====================================================
# === BAGIAN 5: MAIN EXECUTION ===
# ====================================================

if __name__ == "__main__":
    setup_calibration()

    try:
        print("\n" + "="*50)
        print("Aplikasi Terpadu (Sensor + AI) berjalan di:")
        print("Web UI: http://0.0.0.0:5000/")
        print("Data JSON: http://0.0.0.0:5000/data (HANYA UNTUK MENYIMPAN DATA)")
        print("Trigger AI (POST): http://0.0.0.0:5000/trigger_ai")
        print("="*50)
        # Menjalankan Flask di port 5000
        app.run(host="0.0.0.0", port=5000, debug=False)

    except KeyboardInterrupt:
        print("\nStop program...")

    finally:
        # Cleanup
        print("Membersihkan GPIO dan SPI...")
        spi.close()
        GPIO.cleanup()