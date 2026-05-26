from ultralytics import YOLO
import config

model = YOLO(config.MODEL_NAME)

def detect_persons(frame):
    results = model(frame)
    persons = []
    for result in results:
        for box in result.boxes:
            if int(box.cls) == config.PERSON_CLASS_ID:
                conf = float(box.conf[0])
                if conf >= config.CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    persons.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                    })
    return persons
