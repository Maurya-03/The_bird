import cv2
import mediapipe as mp
import time
import os

# ==========================================================
# Setup
# ==========================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils

if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# ==========================================================
# Finger State Logic
# ==========================================================
def get_finger_states(hand, handedness):
    lm = hand.landmark
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb logic
    if handedness == "Right":
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)

    # Other 4 fingers
    for i in range(1, 5):
        fingers.append(lm[tips[i]].y < lm[tips[i] - 2].y)

    return list(map(int, fingers))


# ==========================================================
# Single-Hand Gesture Logic
# ==========================================================
def classify_single(f):
    if f == [0, 0, 1, 0, 0]:
        return "🖕 THE BIRD"
    if f == [0, 1, 0, 0, 1]:
        return "👌 OKAY GESTURE"
    if f == [0, 1, 0, 0, 1]:
        return "👉 DOUBLE POINT"
    if f == [0, 0, 0, 0, 1]:
        return "🤙 CALL ME MAYBE"
    if f == [1, 1, 1, 1, 1]:
        return "✋ TALK TO THE HAND"
    if f == [0, 1, 1, 1, 0]:
        return "🤌 ITALIAN HAND"
    if f == [1, 0, 0, 0, 0]:
        return "🤛 SQUARE UP BRO"
    if f == [0, 1, 0, 0, 0]:
        return "🔫 GUN — PEW PEW"
    if f == [1, 0, 0, 0, 1]:
        return "🏃 I NEED TO GO"

    return "No gesture"


# ==========================================================
# MAIN LOOP
# ==========================================================
cap = cv2.VideoCapture(0)
ptime = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left_text = "Left: None"
    right_text = "Right: None"

    left_f = None
    right_f = None

    gestures = {}  # store per hand gesture

    if results.multi_hand_landmarks:
        for i, hand in enumerate(results.multi_hand_landmarks):

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            handedness = results.multi_handedness[i].classification[0].label
            f = get_finger_states(hand, handedness)

            gesture_name = classify_single(f)
            gestures[handedness] = (f, gesture_name)

            if "THE BIRD" in gesture_name:
                filename = f"screenshots/bird_{int(time.time())}.png"
                cv2.imwrite(filename, frame)

        #==========================================================
        # FIXED KAMEHAMEHA DETECTION (checking by finger states)
        #==========================================================
        if len(gestures) == 2:
            f_left = gestures.get("Left", (None, None))[0]
            f_right = gestures.get("Right", (None, None))[0]

            if f_left == [1,1,1,1,1] and f_right == [1,1,1,1,1]:
                left_text = right_text = "🫸 KAMEHAMEHAAAAAAA !!!"
            else:
                # normal output
                if "Left" in gestures:
                    left_text = f"Left: {gestures['Left'][1]}"
                if "Right" in gestures:
                    right_text = f"Right: {gestures['Right'][1]}"
        else:
            # Only one hand
            for hand_label in gestures:
                if hand_label == "Left":
                    left_text = f"Left: {gestures['Left'][1]}"
                else:
                    right_text = f"Right: {gestures['Right'][1]}"

    # ========================
    # UI Output
    # ========================
    cv2.rectangle(frame, (10, 10), (700, 110), (0, 0, 0), -1)
    cv2.putText(frame, left_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 3)
    cv2.putText(frame, right_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 3)

    # FPS display
    ctime = time.time()
    fps = int(1/(ctime - ptime)) if ptime else 0
    ptime = ctime
    cv2.putText(frame, f"FPS: {fps}", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Advanced Gesture Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
