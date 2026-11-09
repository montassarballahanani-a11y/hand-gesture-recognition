


import cv2
import mediapipe as mp
from pathlib import Path

gestes = ["stop", "left", "right", "ok", "help","none" ]

with mp.solutions.hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.2
) as hands:
    for part in ["train", "val"]:
        for geste in gestes:
            src = Path("dataset_cls") / part / geste
            dst = Path("dataset_hand") / part / geste
            dst.mkdir(parents=True, exist_ok=True)
            n = 0

            for photo in src.glob("*.jpg"):
                img = cv2.imread(str(photo))
                if img is None:
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                if not res.multi_hand_landmarks:
                    continue

                h, w = img.shape[:2]
                pts = res.multi_hand_landmarks[0].landmark
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

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop = cv2.resize(crop, (160, 160))
                cv2.imwrite(str(dst / photo.name), crop)
                n += 1

            print(part, geste, n)
