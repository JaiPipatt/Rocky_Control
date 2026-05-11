import serial
import time
import sys
import struct

# --- CONFIGURATION ---
PORT = 'COM4' 
BAUD = 1000000
CAMERA_INDEX = 1  # Change this to your USB webcam index (0, 1, 2, etc.)

# robot constants for inverse kinematics
WHEEL_RADIUS_A = 0.0297
HALF_WHEELBASE_D = 0.0537
HALF_TRACK_L =  0.0375
RAD_PER_SEC_TO_RPM = 60.0 / (2.0 * 3.14159)

MAX_VELOCITY = 0.4  # Max linear velocity in m/s

# Binary protocol constants
CMD_SET_VELOCITY = 0x01
CMD_STOP = 0x02

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def create_velocity_command(fl_rpm, fr_rpm, rl_rpm, rr_rpm):
    """Create binary command: [0x01][4 floats]"""
    cmd = struct.pack('<Bffff', CMD_SET_VELOCITY, fl_rpm, fr_rpm, rl_rpm, rr_rpm)
    return cmd

def create_stop_command():
    """Create binary command: [0x02]"""
    cmd = struct.pack('<B', CMD_STOP)
    return cmd

print(f"Opening serial port {PORT} at {BAUD} baud...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("Serial port opened successfully.")
except Exception as e:
    print(f"Failed to open port: {e}")
    sys.exit(1)


import cv2
import mediapipe as mp

# Auto-detect camera index
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if cap is None or not cap.isOpened():
    print(f"ERROR: Could not open camera at index {CAMERA_INDEX}.")
    print("Try adjusting CAMERA_INDEX at the top of the script, or close other apps using the camera.")
    sys.exit(1)

print(f"Camera at index {CAMERA_INDEX} opened successfully. Starting hand tracking...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
print("Mediapipe Hands initialized.")

class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.last_error = 0
        self.integral = 0

    def reset(self):
        self.last_error = 0
        self.integral = 0

    def calculate(self, error, dt=0.03): # ~30fps
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.last_error = error
        return output

# Usage
# pid_x = PID(kp=0.5, ki=0.01, kd=0.1)
# velocity_x = pid_x.calculate(error_x)

pid_x = PID(kp=0.6, ki=0.0, kd=0.4)
pid_y = PID(kp=0.4, ki=0.0, kd=0.4)
start_time = time.time()  # Initialize before loop

def ik(forwardBackVel, leftRightVel, rotVel):
    # Desired robot velocities in body frame.
    u = forwardBackVel
    v = leftRightVel
    r = rotVel
    # Wheel angular velocities (rad/s).
    w1 = (1.0 / WHEEL_RADIUS_A) * (u + v + (-HALF_WHEELBASE_D + HALF_TRACK_L) * r)  # front left
    w2 = (1.0 / WHEEL_RADIUS_A) * (u - v + (-HALF_WHEELBASE_D + HALF_TRACK_L) * r)  # front right
    w3 = (1.0 / WHEEL_RADIUS_A) * (u + v + ( HALF_WHEELBASE_D - HALF_TRACK_L) * r)  # rear left
    w4 = (1.0 / WHEEL_RADIUS_A) * (u - v + ( HALF_WHEELBASE_D - HALF_TRACK_L) * r)  # rear right
    # Convert rad/s -> RPM.
    frontLeftRpm  = w1 * RAD_PER_SEC_TO_RPM
    frontRightRpm = w2 * RAD_PER_SEC_TO_RPM
    rearLeftRpm   = w3 * RAD_PER_SEC_TO_RPM
    rearRightRpm  = w4 * RAD_PER_SEC_TO_RPM

    return frontLeftRpm, frontRightRpm, rearLeftRpm, rearRightRpm

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Draw the connections on the hand
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Get coordinates for Landmark 9 (Middle finger MCP - essentially the palm center)
            # Coordinates are normalized (0.0 to 1.0)
            h, w, c = frame.shape
            cx, cy = int(hand_lms.landmark[9].x * w), int(hand_lms.landmark[9].y * h)

            # Calculate error based on deadzone boundaries (not center)
            deadzone_margin = 0.1
            left = w / 2 * (1 - deadzone_margin)
            right = w / 2 * (1 + deadzone_margin)
            top = h / 2 * (1 - deadzone_margin)
            bottom = h / 2 * (1 + deadzone_margin)
            
            # Error is 0 within deadzone, increases as hand moves away from deadzone boundary
            if cx < left:
                error_x = (left - cx) / left
            elif cx > right:
                error_x = -(cx - right) / (w - right)
            else:
                error_x = 0
                
            if cy < top:
                error_y = (top - cy) / top
            elif cy > bottom:
                error_y = -(cy - bottom) / (h - bottom)
            else:
                error_y = 0
            
            if error_x == 0 and error_y == 0:
                pid_x.reset()
                pid_y.reset()
                cmd = create_velocity_command(0.0, 0.0, 0.0, 0.0)
                ser.write(cmd)
                continue
            
            # Use persistent controller state across frames.
            velocity_x = -pid_x.calculate(error_x) #left/right vel
            velocity_y = -pid_y.calculate(error_y) #forward/back vel

            velocity_x = clamp(velocity_x, -MAX_VELOCITY, MAX_VELOCITY)
            velocity_y = clamp(velocity_y, -MAX_VELOCITY, MAX_VELOCITY)


            # Print them to see the "Control Signal" in the terminal
            print(f"Error X: {error_x}, Error Y: {error_y}, Vx: {velocity_x}, Vy: {velocity_y}")

            frontLeftRpm, frontRightRpm, rearLeftRpm, rearRightRpm = ik(velocity_y, velocity_x, 0)  # No rotation control for now
            print(f"Wheel RPMs: FL={frontLeftRpm:.1f}, FR={frontRightRpm:.1f}, RL={rearLeftRpm:.1f}, RR={rearRightRpm:.1f}")    
            cmd = create_velocity_command(frontLeftRpm, frontRightRpm, rearLeftRpm, rearRightRpm)
            ser.write(cmd)
            
            # Highlight the tracking point
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f"Center: {cx}, {cy}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    else:
        pid_x.reset()
        pid_y.reset()
        cmd = create_velocity_command(0.0, 0.0, 0.0, 0.0)
        ser.write(cmd)

    # Draw deadzone box in red (20% from center)
    h, w, c = frame.shape
    deadzone_margin = 0.2  # 20% deadzone
    left = int(w / 2 * (1 - deadzone_margin))
    right = int(w / 2 * (1 + deadzone_margin))
    top = int(h / 2 * (1 - deadzone_margin))
    bottom = int(h / 2 * (1 + deadzone_margin))
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(frame, "Deadzone", (left + 5, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Hand Tracking for Omni-Bot", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Exiting... Sending stop command to robot.")
ser.write(create_stop_command())
pid_x.reset()
pid_y.reset()

cap.release()
cv2.destroyAllWindows()