import numpy as np
import cv2
from model import load_cnn

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cnn = load_cnn('facial_keypoint_model.h5')

cam = cv2.VideoCapture(0)

while(True):
	ret, frame = cam.read()
	frame = cv2.flip(frame, 1)

	grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(grayscale_frame, 1.3, 5)

	for (x, y, w, h) in faces:
		gray_face = grayscale_frame[y:y+h, x:x+h]
		face = frame[y:y+h, x:x+h]

		gray_face = gray_face/255
		gray_face = cv2.resize(gray_face, (96, 96)).reshape(1, 96, 96, 1)

		predicted_keypoints = cnn.predict(gray_face)[0]

		keypoints = []
		for i in range(0, len(predicted_keypoints), 2):
			keypoints.append((int((predicted_keypoints[i]*48+48)*w/96.0), int((predicted_keypoints[i+1]*48+48)*h/96.0)))

		for i in range(len(keypoints)):
			cv2.circle(face, keypoints[i], 5, (0,255,0),5)

		frame[y:y+h, x:x+h] = face

	cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()