import serial, csv, os
from datetime import datetime
from glob import glob

DEFAULT_SERIAL_PORT = '/dev/tty.usbmodemF5PEETZGSCS2R3'
BAUD_RATE = 115200
DEFAULT_DURATION_S = 30
BASE_DIR = 'dataset'

# >>> Stelle das auf deine reale Samplerate ein (Timer / SENSOR_COLLECT_RATE_HZ) <<<
SAMPLE_HZ = 200           # 200 Hz  => 5 ms pro Sample
PERIOD_MS = int(1000 / SAMPLE_HZ)

def pick_capture_type():
    while True:
        c = input("Capture type? [n] normal, [a] anomalie  >>> ").strip().lower()
        if c.startswith('n'):
            return ('normal', os.path.join(BASE_DIR, 'normal_data'))
        if c.startswith('a'):
            return ('anomalie', os.path.join(BASE_DIR, 'anomalie_data'))
        print("Please type 'n' or 'a'.")

def pick_duration(default_s):
    txt = input(f"Capture duration in seconds [{default_s}] >>> ").strip()
    if not txt:
        return default_s
    try:
        return max(1, int(txt))
    except ValueError:
        print("Invalid number, using default.")
        return default_s

def open_serial(port_hint):
    try:
        return serial.Serial(port_hint, BAUD_RATE, timeout=1)
    except Exception:
        candidates = sorted(glob('/dev/tty.usbmodemROJP4ADDBGNDS3') + glob('/dev/ttyUSB*'))
        if not candidates:
            raise RuntimeError(f"Could not open {port_hint} and found no /dev/ttyACM* or /dev/ttyUSB* devices.")
        print(f"[Info] Using detected port: {candidates[0]}")
        return serial.Serial(candidates[0], BAUD_RATE, timeout=1)

def fix_twos_complement(n):
    n = int(n)
    if 0 <= n <= 0xFFFF:
        return n - 0x10000 if n >= 0x8000 else n  # int16
    if n >= 0x80000000:
        return n - 0x100000000                    # int32
    return n

def main():
    label, out_dir = pick_capture_type()
    duration_s = pick_duration(DEFAULT_DURATION_S)
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    csv_path = os.path.join(out_dir, f"{label}_log_{ts}.csv")

    ser = open_serial(DEFAULT_SERIAL_PORT)
    print(f"\nConnected to {ser.port} @ {BAUD_RATE} baud")
    print(f"Logging '{label}' data for {duration_s}s -> {csv_path}\n")

    max_samples = duration_s * SAMPLE_HZ
    sample_idx = 0

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "ax_raw", "ay_raw", "az_raw", "label"])

        try:
            while sample_idx < max_samples:
                line = ser.readline().decode(errors='ignore').strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) != 3:
                    continue
                try:
                    x = fix_twos_complement(parts[0])
                    y = fix_twos_complement(parts[1])
                    z = fix_twos_complement(parts[2])
                except ValueError:
                    continue

                t_ms = sample_idx * PERIOD_MS
                writer.writerow([t_ms, x, y, z, label])
                if (sample_idx % SAMPLE_HZ) == 0:
                    # einmal pro Sekunde eine Zeile in der Konsole
                    print(f"{t_ms:6d} ms -> {x:6d},{y:6d},{z:6d}  [{label}]")
                sample_idx += 1
        except KeyboardInterrupt:
            print("\n[Interrupted] Stopping early…")
        finally:
            ser.close()

    print(f"\n Done! Data saved to: {csv_path}")

if __name__ == "__main__":
    main()
