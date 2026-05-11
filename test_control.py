import serial
import time
import sys

# --- CONFIGURATION ---
PORT = 'COM4' 
BAUD = 1000000

print(f"Opening serial port {PORT} at {BAUD} baud...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("Serial port opened successfully.")
except Exception as e:
    print(f"Failed to open port: {e}")
    sys.exit(1)

# Wait briefly for serial connection to stabilize
time.sleep(1)

# Hardcoded test command: Forward at 0.1 m/s (vx, vy, wz)
print("Sending test command: Forward at 0.1 m/s...")
cmd = "CMD,0.100,0.000,0.000\n"

print("Running motors forward for 2 seconds...")
start_time = time.time()

# We MUST loop to beat the 250ms timeout in the OpenCR code
while time.time() - start_time < 2.0:
    ser.write(cmd.encode('ascii'))
    time.sleep(0.1)  # Send every 100ms

print("Stopping robot...")
ser.write(b"STOP\n")
time.sleep(0.1) # Give the stop command time to transmit
ser.close()

print("Test complete.")