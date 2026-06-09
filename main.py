import threading
import cv2
import config
from detector import detect_persons
from face_recognizer import load_known_faces, recognize_faces
from dwell_time import DwellTimeTracker
from visualization import draw_person
from report import load_report, add_session, print_summary


class VideoCaptureThread:
    def __init__(self, camera_id):
        self._cap = cv2.VideoCapture(camera_id)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
        return False, None

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


def main():
    print("Carregando faces conhecidas...")
    load_known_faces()

    report = load_report()
    tracker = DwellTimeTracker()
    cap = VideoCaptureThread(config.CAMERA_ID)

    last_persons = []
    last_faces = []
    frame_count = 0

    print("Pressione 'q' para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display_frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        if frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(frame, (320, 240))
            raw_persons = detect_persons(small_frame)
            raw_faces = recognize_faces(small_frame)

            sx = config.FRAME_WIDTH / 320
            sy = config.FRAME_HEIGHT / 240

            last_persons = []
            for p in raw_persons:
                x1, y1, x2, y2 = p['bbox']
                last_persons.append({
                    'bbox': (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)),
                    'confidence': p['confidence'],
                })

            last_faces = []
            for f in raw_faces:
                ftop, fright, fbottom, fleft = f['bbox']
                last_faces.append({
                    'bbox': (int(ftop * sy), int(fright * sx), int(fbottom * sy), int(fleft * sx)),
                    'name': f['name'],
                })

            names_in_frame = []
            for person in last_persons:
                best_name = "Unknown"
                for face in last_faces:
                    x1, y1, x2, y2 = person['bbox']
                    ftop, fright, fbottom, fleft = face['bbox']
                    fx = (fleft + fright) // 2
                    fy = (ftop + fbottom) // 2
                    if x1 <= fx <= x2 and y1 <= fy <= y2:
                        best_name = face['name']
                        break

                if best_name != "Unknown":
                    names_in_frame.append(best_name)
                dwell = tracker.get_dwell_time(best_name)
                draw_person(display_frame, person['bbox'], best_name, person['confidence'], dwell)

            departed = tracker.update(names_in_frame)
            for name, duration, start_ts, end_ts in departed:
                session = add_session(report, name, duration, start_ts, end_ts)
                print(f"  {session['name']} saiu: {session['duration_seconds']:.1f}s")
        else:
            for person in last_persons:
                best_name = "Unknown"
                for face in last_faces:
                    x1, y1, x2, y2 = person['bbox']
                    ftop, fright, fbottom, fleft = face['bbox']
                    fx = (fleft + fright) // 2
                    fy = (ftop + fbottom) // 2
                    if x1 <= fx <= x2 and y1 <= fy <= y2:
                        best_name = face['name']
                        break
                dwell = tracker.get_dwell_time(best_name)
                draw_person(display_frame, person['bbox'], best_name, person['confidence'], dwell)

        cv2.imshow('YOLOv8 + Face Recognition + Dwell Time', display_frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    remaining = tracker.flush()
    for name, duration, start_ts, end_ts in remaining:
        add_session(report, name, duration, start_ts, end_ts)

    cap.release()
    cv2.destroyAllWindows()

    print_summary(report)


if __name__ == '__main__':
    main()
