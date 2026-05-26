import cv2
import config
from detector import detect_persons
from face_recognizer import load_known_faces, recognize_faces
from dwell_time import DwellTimeTracker
from visualization import draw_person

def main():
    print("Carregando faces conhecidas...")
    load_known_faces()

    tracker = DwellTimeTracker()
    cap = cv2.VideoCapture(config.CAMERA_ID)

    print("Pressione 'q' para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        persons = detect_persons(frame)
        faces = recognize_faces(frame)

        names_in_frame = []
        for person in persons:
            x1, y1, x2, y2 = person['bbox']
            best_name = "Unknown"
            for face in faces:
                ftop, fright, fbottom, fleft = face['bbox']
                fx = (fleft + fright) // 2
                fy = (ftop + fbottom) // 2
                if x1 <= fx <= x2 and y1 <= fy <= y2:
                    best_name = face['name']
                    break

            names_in_frame.append(best_name)
            dwell = tracker.get_dwell_time(best_name)
            draw_person(frame, person['bbox'], best_name, person['confidence'], dwell)

        tracker.update(names_in_frame)

        cv2.imshow('YOLOv8 + Face Recognition + Dwell Time', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
