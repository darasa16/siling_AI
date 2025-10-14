import requests
import time
import statistics

# === KONFIGURASI ===
HEARTBEAT_URL = "https://intertown-velia-ichthyologic.ngrok-free.dev/heartbeat"
DOWNLOAD_URL = "https://intertown-velia-ichthyologic.ngrok-free.dev/download_test"  # endpoint file dummy
NUM_REQUESTS = 50  # jumlah request uji latency / packet loss
DOWNLOAD_SIZE_MB = 5  # ukuran file dummy dalam MB

# ===================
# Fungsi Latency / Packet Loss / Jitter
latencies = []

print(f"Mulai uji latency {NUM_REQUESTS} request ke {HEARTBEAT_URL}...")
for i in range(NUM_REQUESTS):
    try:
        start = time.time()
        r = requests.get(HEARTBEAT_URL, timeout=5)
        elapsed_ms = (time.time() - start) * 1000
        latencies.append(elapsed_ms)
        print(f"[{i+1}] Status: {r.status_code}, Latency: {elapsed_ms:.2f} ms")
    except:
        latencies.append(None)
        print(f"[{i+1}] Request gagal (timeout / error)")

# Analisis latency
received = [x for x in latencies if x is not None]
lost = len(latencies) - len(received)
packet_loss_percent = (lost / NUM_REQUESTS) * 100
jitter = statistics.stdev(received) if len(received) > 1 else 0

print("\n=== Hasil Latency / Packet Loss / Jitter ===")
print(f"Packet loss: {packet_loss_percent:.2f}%")
if received:
    print(f"Latency min: {min(received):.2f} ms")
    print(f"Latency avg: {sum(received)/len(received):.2f} ms")
    print(f"Latency max: {max(received):.2f} ms")
    print(f"Jitter (std dev): {jitter:.2f} ms")
else:
    print("Semua request gagal.")

# ===================
# Uji Throughput / Bandwidth
print(f"\nMulai uji throughput dengan mengunduh file dummy {DOWNLOAD_SIZE_MB} MB...")
start = time.time()
try:
    r = requests.get(DOWNLOAD_URL, timeout=30)
    elapsed_sec = time.time() - start
    throughput_mbps = (DOWNLOAD_SIZE_MB * 8) / elapsed_sec
    print(f"Download selesai dalam {elapsed_sec:.2f} s")
    print(f"Perkiraan throughput: {throughput_mbps:.2f} Mbps")
except:
    print("Download gagal / timeout")
