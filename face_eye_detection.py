#https://www.youtube.com/watch?v=mPCZLOVTEc4&list=PLzMcBGfZo4-lUA8uGjeXhBUUzPYc6vZRn&index=8

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_casecade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_casecade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


while True:

	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_casecade.detectMultiScale(gray, 1.3, 5)

	for (x, y, w, h) in faces:

		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
		#print("x = ", x, "\n", "y = ", y, "\n", "w = ", w, "\n", "h = ", h)

		reigon_off_intreset_gray = gray[y : y + w, x : x + h]
		reigon_off_intreset_color = frame[y : y + h, x : x + w]
		eyes = eye_casecade.detectMultiScale(reigon_off_intreset_gray, 1.3, 5)

		for (ex , ey, ew, eh) in eyes:
			cv2.rectangle(reigon_off_intreset_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)

	#print(faces, "\n")

	cv2.imshow("image", frame)

	if cv2.waitKey(1) == ord('q') :
		break

cap.release()
cv2.destroyAllWindows()