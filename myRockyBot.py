import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # important on Windows



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.last_error = 0
        self.integral = 0

    def calculate(self, error, dt=0.03): # ~30fps
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.last_error = error
        return output

# Usage in your loop:
# pid_x = PID(kp=0.5, ki=0.01, kd=0.1)
# velocity_x = pid_x.calculate(error_x)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
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
            
            # Assuming your camera is 640x480
            error_x = ( (w / 2) - cx ) / (w / 2)  # Horizontal error
            error_y = ( (h / 2) - cy ) / (h / 2)  # Vertical error
            
            if abs(error_x) < 0.05 and abs(error_y) < 0.05: # 5% deadzone
                error_x = 0
                error_y = 0

            # PIDs
            pid_x = PID(kp=0.5, ki=0.0, kd=0.0)
            velocity_x = -pid_x.calculate(error_x)

            pid_y = PID(kp=0.5, ki=0.0, kd=0.0)
            velocity_y = -pid_y.calculate(error_y)

            # Print them to see the "Control Signal" in the terminal
            print(f"Error X: {error_x}, Error Y: {error_y}, Vx: {velocity_x}, Vy: {velocity_y}")
            
            # Highlight the tracking point
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f"Center: {cx}, {cy}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking for Omni-Bot", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()