import numpy as np
import cv2
from model import load_cnn

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cnn = load_cnn('facial_keypoint_model.h5')

cam = cv2.VideoCapture(0)

while(True):
	ret, frame = cam.read()
	frame = cv2.flip(frame, 1) # Main frame
	frame2 = frame.copy() # Smaller frame to visualize the keypoint predictions

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)

	for (x, y, w, h) in faces:
		face = frame[y:y+h, x:x+h]
		face2 = frame2[y:y+h, x:x+h]

		gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) # Pass in grayscale face to model
		gray_face = gray_face/255.0
		gray_face = cv2.resize(gray_face, (96, 96)).reshape(1, 96, 96, 1)

		predicted_keypoints = cnn.predict(gray_face)[0]

		keypoints = []
		for i in range(0, len(predicted_keypoints), 2): # During training, keypoints were scaled to range [-1, 1] - so descale data here
			keypoints.append((int((predicted_keypoints[i]*48+48)*w/96.0), int((predicted_keypoints[i+1]*48+48)*h/96.0)))

		# Glasses filter code
		glasses = cv2.imread('images/glasses.png', -1)
		glasses_width = int((keypoints[3][0] - keypoints[5][0])*2)
		glasses_height = int((keypoints[0][1] - keypoints[6][1])*3.7)
		glasses = cv2.resize(glasses, (glasses_width, glasses_height))
		y1, y2 = keypoints[0][1]+int(w*0.1)-glasses_height, keypoints[0][1]+int(w*0.1)
		x1, x2 = keypoints[5][0]-int(w*0.23), keypoints[5][0]+glasses_width-int(w*0.23)

		# Handle transparency of glasses filter
		alpha = glasses[:,:,3]/255.0
		alpha_face = 1 - alpha
		for c in range(0, 3):
			min_y = min(w,y2)
			min_x = min(h,x2)
			face[y1:min_y, x1:min_x, c] = (alpha[:min_y,:min_x] * glasses[:min_y,:min_x,c] +
				alpha_face[:min_y,:min_x]*face[y1:min_y, x1:min_x, c])

		# Map modified face back to frame
		frame[y:y+h, x:x+h] = face

		# Draw out keypoints on smaller frame
		for i in range(len(keypoints)):
			cv2.circle(face2, keypoints[i], 5, (0,255,0), 5)

	frame2 = cv2.resize(frame2, (int(frame.shape[1]*0.3),int(frame.shape[0]*0.3)))
	y1, y2 = int(frame.shape[0]*0.65), int(frame.shape[0]*0.65) + frame2.shape[0]
	x1, x2 = int(frame.shape[1]*0.65), int(frame.shape[1]*0.65) + frame2.shape[1]
	frame[y1:y2, x1:x2,:] = frame2[:,:,:]

	cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()