import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)
raka_image = face_recognition.load_image_file("faces/raka.jpg")
raka_face_encoding = face_recognition.face_encodings(raka_image)[0]
likhitha_image = face_recognition.load_image_file("faces/likhitha.jpg")
likhitha_face_encoding = face_recognition.face_encodings(likhitha_image)[0]
rohith_image = face_recognition.load_image_file("faces/rohith.jpg")
rohith_face_encoding = face_recognition.face_encodings(rohith_image)[0]

known_face_encodings = [raka_face_encoding, likhitha_face_encoding, rohith_face_encoding]   # List of known face encodings
known_face_names = ["Raka", "Likhitha", "Rohith"]   # List of known face names 

students = known_face_names.copy()
face_locations = []
face_encodings = []

now = datetime.now()
current_date = now.strftime("%d/%m/%Y")

f=open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # for face_encoding in face_encodings:
    #     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
       

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distance)
    if(matches[best_match_index]):
        name = known_face_names[best_match_index]


while True:
    cv2.imshow("Attendace", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
