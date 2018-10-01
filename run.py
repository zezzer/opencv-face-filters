import numpy as np
import cv2
from model import load_cnn

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cnn = load_cnn('facial_keypoint_model.h5')
filter_num = 0

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

		img = []
		img_height = 0
		img_width = 0

		if filter_num == 0: # Glasses Filter
			img = cv2.imread('images/glasses.png', -1)
			img_width = int((keypoints[3][0] - keypoints[5][0])*2)
			img_height = int((keypoints[0][1] - keypoints[6][1])*3.7)
			img = cv2.resize(img, (img_width, img_height))
			y1, y2 = keypoints[0][1]+int(w*0.13)-img_height, keypoints[0][1]+int(w*0.13)
			x1, x2 = keypoints[5][0]-int(w*0.23), keypoints[5][0]+img_width-int(w*0.23)

		elif filter_num == 1:  # Mustache Filter
			img = cv2.imread('images/mustache.png', -1)
			img_width = int((keypoints[11][0] - keypoints[12][0])*3.5)
			img_height = int((keypoints[14][1] - keypoints[13][1])*3)
			img = cv2.resize(img, (img_width, img_height))
			y1, y2 = keypoints[13][1]-int(w*0.13), keypoints[13][1]+img_height-int(w*0.13)
			x1, x2 = keypoints[12][0]-int(w*0.33), keypoints[12][0]+img_width-int(w*0.33)

		elif filter_num == 2:  # Cat Whiskers Filter
			img = cv2.imread('images/whiskers.png', -1)
			img_width = int((keypoints[3][0] - keypoints[5][0])*2)
			img_height = int((keypoints[13][1] - keypoints[10][1])*7)
			img = cv2.resize(img, (img_width, img_height))
			y1, y2 = keypoints[10][1]-int(w*0.43), keypoints[10][1]+img_height-int(w*0.43)
			x1, x2 = keypoints[5][0]-int(w*0.22), keypoints[5][0]+img_width-int(w*0.22)

		# Handle transparency of filter
		alpha = img[:,:,3]/255.0
		alpha_face = 1 - alpha
		for c in range(0, 3):
			min_y = min(h-y1,img_height)
			min_x = min(w-x1,img_width)
			face[y1:min(h,y2), x1:min(w,x2), c] = (alpha[:min_y,:min_x] * img[:min_y,:min_x,c] +
				alpha_face[:min_y,:min_x]*face[y1:min(h,y2), x1:min(w,x2), c])

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

	c = cv2.waitKey(1)
	if c == ord('q'):
		break
	elif c == ord(' '):
		filter_num = (filter_num+1)%3

cam.release()
cv2.destroyAllWindows()