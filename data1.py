# ====================================================
# === BAGIAN JARINGAN & SISTEM: IMPORT & SETUP ===
# ====================================================

import time
from flask import Flask, jsonify
import socket, requests, psutil, subprocess
from datetime import datetime 
import csv 
import os

# --- Variabel Global untuk Metrik ---
app = Flask(__name__)
request_count = 0
start_time = time.time()
try:
    net_counters = psutil.net_io_counters()
    last_bytes_sent = net_counters.bytes_sent
    last_bytes_recv = net_counters.bytes_recv
    last_time = time.time()
except Exception:
    last_bytes_sent, last_bytes_recv, last_time = 0, 0, time.time()

# 🌟 SETUP LOG CSV (Hanya untuk Jaringan/Sistem)
NETWORK_LOG_FILE = "network_system_log.csv"

NETWORK_FIELDS = [
    'timestamp', 'server_latency_ms', 
    'tx_speed_kbs', 'rx_speed_kbs', 'latency_ping_ms', 
    'cpu_usage_percent', 'mem_usage_percent',
    'ip_local', 'ip_public', 'ping_status'
]

def init_csv_log(filename, fieldnames):
    """Membuat file CSV jika belum ada dan menulis header."""
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    if not os.path.exists(filename):
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            print(f"✅ File log {filename} berhasil dibuat.")
        except Exception as e:
            print(f"❌ Gagal membuat file {filename}: {e}")

# ====================================================
# === BAGIAN FUNGSI UTILITY JARINGAN & LOGGING ===
# ====================================================

def system_and_network_analysis():
    """Mengumpulkan metrik sistem (CPU, RAM) dan jaringan."""
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
    
    # Metrik Sistem
    cpu_percent = psutil.cpu_percent(interval=None)
    mem_info = psutil.virtual_memory()

    now = time.time()
    elapsed = now - last_time
    
    # Hitung kecepatan transfer dalam KB/s
    if elapsed > 0.01:
        tx_speed = (bytes_sent - last_bytes_sent) / elapsed / 1024
        rx_speed = (bytes_recv - last_bytes_recv) / elapsed / 1024
    else:
        tx_speed, rx_speed = 0.0, 0.0

    last_bytes_sent, last_bytes_recv, last_time = bytes_sent, bytes_recv, now

    # Ping Latency
    latency = "Timeout"
    ping_status = "Timeout"
    try:
        ping = subprocess.check_output(
            ["ping", "-c", "1", "-W", "1", "8.8.8.8"],
            universal_newlines=True
        )
        latency_line = [line for line in ping.split('\n') if 'avg' in line]
        if latency_line:
            latency_ms = latency_line[0].split('/')[4]
            latency = f"{latency_ms} ms"
            ping_status = "OK"
        else:
            latency = "Timeout"
    except:
        pass

    return {
        "ip_local": ip_local,
        "ip_public": ip_public,
        "tx_speed": f"{tx_speed:.2f} KB/s",
        "rx_speed": f"{rx_speed:.2f} KB/s",
        "latency": latency,
        "ping_status": ping_status,
        "cpu_usage": f"{cpu_percent:.1f}%",
        "mem_usage": f"{mem_info.percent:.1f}%",
        "tx_total": f"{bytes_sent/1024/1024:.2f} MB",
        "rx_total": f"{bytes_recv/1024/1024:.2f} MB",
        "interfaces": {name: stats.isup for name, stats in interfaces.items()}
    }

def save_network_log(net_info, processing_time_ms):
    """Menyimpan data jaringan, sistem, dan latensi internal ke CSV."""
    
    try:
        with open(NETWORK_LOG_FILE, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=NETWORK_FIELDS)
            
            # Konversi data string ke numerik untuk CSV
            tx_speed_val = float(net_info.get('tx_speed', '0.0 KB/s').split(' ')[0])
            rx_speed_val = float(net_info.get('rx_speed', '0.0 KB/s').split(' ')[0])
            
            ping_latency_str = net_info.get('latency', '0.0 ms')
            latency_ping_val = float(ping_latency_str.split(' ')[0]) if 'Timeout' not in ping_latency_str else None
            
            cpu_usage_val = float(net_info.get('cpu_usage', '0.0%').split('%')[0])
            mem_usage_val = float(net_info.get('mem_usage', '0.0%').split('%')[0])

            log_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                'server_latency_ms': processing_time_ms,
                'tx_speed_kbs': tx_speed_val,
                'rx_speed_kbs': rx_speed_val,
                'latency_ping_ms': latency_ping_val,
                'cpu_usage_percent': cpu_usage_val,
                'mem_usage_percent': mem_usage_val,
                'ip_local': net_info.get('ip_local'),
                'ip_public': net_info.get('ip_public'),
                'ping_status': net_info.get('ping_status'),
            }
            
            writer.writerow(log_data)
            print(f"💾 Log jaringan/sistem berhasil disimpan ke {NETWORK_LOG_FILE}.")
            
    except Exception as e:
        print(f"❌ Gagal menulis log jaringan ke CSV: {e}")

# ====================================================
# === BAGIAN FLASK ENDPOINTS JARINGAN ===
# ====================================================

@app.route("/data")
def get_network_json():
    """Endpoint untuk AJAX, mengembalikan data jaringan & sistem dalam JSON, dan log ke CSV."""
    global request_count, start_time
    
    # 🌟 Metrik 1: Hitung Throughput Global
    request_count += 1
    elapsed = time.time() - start_time
    throughput = request_count / elapsed if elapsed > 0 else 0
    
    # 🌟 Metrik 2: Pengukuran Latensi Internal (Processing Time)
    start_proc_time = time.time()
    
    net_info = system_and_network_analysis()
    
    end_proc_time = time.time()
    processing_time_ms = (end_proc_time - start_proc_time) * 1000
    
    # 🌟 Logging CSV: Simpan metrik jaringan dan sistem
    save_network_log(net_info, processing_time_ms)
    
    # Log ke Terminal
    print("=" * 40)
    print(f"Request: {request_count}, Throughput: {throughput:.2f} req/s")
    print(f"CPU/RAM: {net_info['cpu_usage']} / {net_info['mem_usage']}")
    print(f"TX/RX Speed: {net_info['tx_speed']} / {net_info['rx_speed']}, Ping: {net_info['latency']}")
    print(f"Internal Latency /data: {processing_time_ms:.3f} ms")
    print("=" * 40)
    
    # Siapkan JSON Response
    response_data = {
        **net_info,
        "server_processing_time_ms": f"{processing_time_ms:.3f} ms",
        "global_throughput": f"{throughput:.2f} req/s"
    }
    
    return jsonify(response_data)


# ====================================================
# === BAGIAN MAIN EXECUTION ===
# ====================================================

if __name__ == "__main__":
    
    # Inisialisasi file CSV
    init_csv_log(NETWORK_LOG_FILE, NETWORK_FIELDS)

    try:
        print("\n" + "="*50)
        print("Aplikasi Uji Jaringan Berjalan di:")
        print("Data JSON (Log Latency/Network): http://0.0.0.0:5000/data")
        print("Log CSV Network/System: network_system_log.csv")
        print("="*50)
        
        # Jalankan Flask
        app.run(host="0.0.0.0", port=5000, debug=False)

    except KeyboardInterrupt:
        print("\nStop program...")

    finally:
        # Cleanup (jika ada, tetapi di sini hanya mencetak pesan)
        print("Program selesai.")