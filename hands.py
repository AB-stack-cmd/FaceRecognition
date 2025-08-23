import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Webcam
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract key points
            landmarks = hand_landmarks.landmark
            index_finger = landmarks[8]   # Index fingertip
            thumb_finger = landmarks[4]   # Thumb tip
            middle_finger = landmarks[12] # Middle fingertip

            # Convert to screen coords
            x = int(index_finger.x * screen_w)
            y = int(index_finger.y * screen_h)

            # Move mouse with index finger
            pyautogui.moveTo(x, y)

            # Calculate distance between index and thumb → "Click"
            dist = math.hypot(
                (index_finger.x - thumb_finger.x),
                (index_finger.y - thumb_finger.y)
            )

            # if dist < 0.02:  # Small distance = Click``
            #     pyautogui.click()

            # Zoom (distance between thumb & middle finger)
            zoom_dist = math.hypot(
                (middle_finger.x - thumb_finger.x),
                (middle_finger.y - thumb_finger.y)
            )

            if dist <0.01:
                pyautogui.click()

            # if zoom_dist < 0.05:  # Pinch close = Zoom out
            #     pyautogui.hotkey("ctrl", "-")
            # elif zoom_dist > 0.12:  # Fingers apart = Zoom in
            #     pyautogui.hotkey("ctrl", "+")

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
