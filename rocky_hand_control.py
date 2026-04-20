import time

import cv2
import mediapipe as mp
import serial
from serial.tools import list_ports


MAX_VX = 0.30
MAX_VY = 0.30
MAX_WZ = 0.0
DEADZONE = 0.05
SEND_PERIOD_S = 0.05  # 20 Hz, safely faster than OpenCR timeout (250 ms)

SERIAL_PORT = "auto"  # Set e.g. "/dev/ttyACM0" if auto detect is wrong.
SERIAL_BAUD = 1_000_000
CAMERA_INDEX = 0


class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_error = 0.0
        self.integral = 0.0

    def calculate(self, error, dt):
        dt = max(dt, 1e-3)
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.last_error = error
        return output


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def auto_find_opencr_port():
    candidates = []
    for p in list_ports.comports():
        text = f"{p.device} {p.description} {p.hwid}".lower()
        if "opencr" in text or "acm" in text or "usb" in text:
            candidates.append(p.device)

    if candidates:
        return candidates[0]

    # Common Raspberry Pi USB serial names.
    for fallback in ("/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0"):
        candidates.append(fallback)

    return candidates[0]


class OpenCRSerial:
    def __init__(self, port, baudrate):
        self.serial_port = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=0.0,
            write_timeout=0.05,
        )

        # OpenCR may reset on USB open.
        time.sleep(1.2)
        self.serial_port.reset_input_buffer()
        self.serial_port.reset_output_buffer()

    def send_cmd(self, vx, vy, wz):
        line = f"CMD,{vx:.3f},{vy:.3f},{wz:.3f}\n"
        self.serial_port.write(line.encode("ascii"))

    def send_stop(self):
        self.serial_port.write(b"STOP\n")

    def close(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()


def main():
    serial_port_name = auto_find_opencr_port() if SERIAL_PORT == "auto" else SERIAL_PORT
    print(f"[INFO] Opening OpenCR serial: {serial_port_name} @ {SERIAL_BAUD}")

    try:
        opencr = OpenCRSerial(serial_port_name, SERIAL_BAUD)
    except serial.SerialException as exc:
        print(f"[ERROR] Cannot open serial port {serial_port_name}: {exc}")
        print("[INFO] Available ports:")
        for p in list_ports.comports():
            print(f"  - {p.device} ({p.description})")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam index {CAMERA_INDEX}")
        opencr.send_stop()
        opencr.close()
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    pid_x = PID(kp=0.8, ki=0.0, kd=0.05)
    pid_y = PID(kp=0.8, ki=0.0, kd=0.05)

    last_send = 0.0
    last_t = time.time()

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        ) as hands:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("[WARN] Webcam frame grab failed")
                    break

                now = time.time()
                dt = now - last_t
                last_t = now

                frame = cv2.flip(frame, 1)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                target_vx = 0.0
                target_vy = 0.0
                target_wz = 0.0
                hand_detected = False

                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]
                    hand_detected = True

                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                    h, w, _ = frame.shape
                    cx = int(hand_lms.landmark[9].x * w)
                    cy = int(hand_lms.landmark[9].y * h)

                    error_x = ((w / 2.0) - cx) / (w / 2.0)
                    error_y = ((h / 2.0) - cy) / (h / 2.0)

                    if abs(error_x) < DEADZONE:
                        error_x = 0.0
                    if abs(error_y) < DEADZONE:
                        error_y = 0.0

                    # Map image errors to body velocities.
                    # error_y -> forward/back (vx), error_x -> left/right (vy)
                    target_vx = clamp(pid_y.calculate(error_y, dt) * MAX_VX, -MAX_VX, MAX_VX)
                    target_vy = clamp(pid_x.calculate(error_x, dt) * MAX_VY, -MAX_VY, MAX_VY)
                    target_wz = clamp(target_wz, -MAX_WZ, MAX_WZ)

                    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), cv2.FILLED)
                    cv2.putText(
                        frame,
                        f"ErrX:{error_x:+.2f} ErrY:{error_y:+.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                if now - last_send >= SEND_PERIOD_S:
                    if hand_detected:
                        opencr.send_cmd(target_vx, target_vy, target_wz)
                        print(
                            f"CMD,{target_vx:+.3f},{target_vy:+.3f},{target_wz:+.3f}",
                            flush=True,
                        )
                    else:
                        opencr.send_stop()
                        print("STOP", flush=True)
                    last_send = now

                mode = "TRACK" if hand_detected else "NO HAND"
                cv2.putText(
                    frame,
                    f"{mode} | Vx:{target_vx:+.2f} Vy:{target_vy:+.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 220, 0),
                    2,
                )

                cv2.imshow("Rocky Hand Control", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        try:
            opencr.send_stop()
        except serial.SerialException:
            pass
        opencr.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()