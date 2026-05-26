import os
import face_recognition
import numpy as np
import config

known_face_encodings = []
known_face_names = []

def load_known_faces():
    known_face_encodings.clear()
    known_face_names.clear()
    for filename in os.listdir(config.KNOWN_FACES_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
            image_path = os.path.join(config.KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)

def recognize_faces(frame):
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        return []

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        results.append({
            'bbox': (top, right, bottom, left),
            'name': name,
        })
    return results
