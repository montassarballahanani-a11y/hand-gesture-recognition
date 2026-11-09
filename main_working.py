import cv2
import mediapipe as mp
from ultralytics import YOLO

model = YOLO("runs/classify/gestes_main_v2/weights/best.pt")

hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.3
)

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise SystemExit("Camera indisponible")

while True:
    ok, frame = cam.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = "NONE"

    if result.multi_hand_landmarks:
        h, w = frame.shape[:2]
        pts = result.multi_hand_landmarks[0].landmark

        xs = [p.x * w for p in pts]
        ys = [p.y * h for p in pts]

        cx = int((min(xs) + max(xs)) / 2)
        cy = int((min(ys) + max(ys)) / 2)

        s = int(max(max(xs)-min(xs),
                    max(ys)-min(ys)) * 1.5)

        x1 = max(0, cx-s//2)
        x2 = min(w, cx+s//2)
        y1 = max(0, cy-s//2)
        y2 = min(h, cy+s//2)

        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            crop = cv2.resize(crop, (160, 160))
            pred = model.predict(
                crop, device="cpu",
                verbose=False
            )[0]

            confiance = float(pred.probs.top1conf)
            label = pred.names[pred.probs.top1]
            if label in ("left", "right"):
                dx = pts[8].x - pts[5].x
                if abs(dx) > 0.12:
                    label = "left" if dx > 0 else "right"

    cv2.putText(frame, label.upper(), (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
hands.close()
cv2.destroyAllWindows()

