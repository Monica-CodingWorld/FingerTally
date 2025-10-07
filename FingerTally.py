import cv2
import mediapipe as mp

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# odd or even
def odd_or_even(fingers):
    return "Even" if fingers % 2 == 0 else "Odd"    

# Function to count extended fingers
def count_fingers(landmarks):
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_mcp = [2, 5, 9, 13, 17]    # Base joints of fingers

    count = 0
    for tip, mcp in zip(finger_tips, finger_mcp):
        if landmarks[tip].y < landmarks[mcp].y:  # If tip is above MCP, it's extended(y co-ordinate)
            count += 1

    return count

# Function to determine left or right hand
def detect_hand(landmarks):
    if landmarks[17].x < landmarks[5].x:
        return "Right Hand"
    else:
        return "Left Hand"

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for a natural view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for hand tracking
    result = hands.process(rgb_frame)
    total_fingers = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand mesh
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert landmarks to a list
            landmarks = hand_landmarks.landmark
            finger_count = count_fingers(landmarks)
            total_fingers += finger_count

            # Detect hand side
            hand_side = detect_hand(landmarks)

            # Display hand side and fingers count
            cv2.putText(frame, f"{hand_side}: {finger_count}/5", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Changed to red(as suggested while review)

    # Show total fingers count (out of 10)
    result_text = f"Total Fingers: {total_fingers}/10 ({odd_or_even(total_fingers)})"
    cv2.putText(frame, result_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Changed to red

    # Show the video feed
    cv2.imshow("Hand Mesh Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

