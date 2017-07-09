# USAGE 1:
# blabla

# USAGE 2:
# blabla

# ------------------------------------------------------------------------------
# import necessary libraries
import numpy as np
import cv2
import argparse
import imutils
from timeit import default_timer as timer

# ------------------------------------------------------------------------------
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='path to the (optional) video file')
ap.add_argument('-i', '--image', help='path to the (optional) image file')
args = vars(ap.parse_args())

## __error must be raised if both image and video is given__

# ------------------------------------------------------------------------------
# grab input based on given arguments
if args.get('image'):
	img = cv2.imread(args["image"])
	print('# INPUT: USER GIVEN IMAGE')
elif not args.get('video'):
	camera = cv2.VideoCapture(0)
	print('# INPUT: WEBCAM VIDEO STREAM')
else:
	camera = cv2.VideoCapture(args['video'])
	print('# INPUT: USER GIVEN VIDEO')
print('-------------------------------')


# ------------------------------------------------------------------------------
# function for processing a single image
# def processViolaJones(image,face_cascade,eye_cascade):
# 	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# 	faces = face_cascade.detectMultiScale(gray, 1.3, 5) # __RANDOM HYPERPARAMETERS HERE__
# 	for (x,y,w,h) in faces:
# 	    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
# 	    roi_gray = gray[y:y+h, x:x+w]
# 	    roi_color = image[y:y+h, x:x+w]
# 	    eyes = eye_cascade.detectMultiScale(roi_gray)
# 	    for (ex,ey,ew,eh) in eyes:
# 	        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# 	return image

# ------------------------------------------------------------------------------
# function for processing a single image
# def applyHaarCascade(image,haarcascade,r,g,b):

# 	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# 	detected_regions = haarcascade.detectMultiScale(gray)

# 	for (x,y,w,h) in detected_regions:
# 	    cv2.rectangle(image,(x,y),(x+w,y+h),(b,g,r),2)

# 	return image, detected_regions

class FoundSubFeature:
	def __init__(self,parent,subf_x,subf_y,subf_w,subf_h,face_x,face_y,face_w,face_h):
		self.parent = parent
		self.midpoint = [subf_x+subf_w/2,subf_y+subf_h/2]

		face_midpoint = [face_x+face_w/2,face_y+face_h/2]
		self.relative_midpoint = [self.midpoint[0]-face_midpoint[0],self.midpoint[1]-face_midpoint[1]]

		self.bounding_box = [subf_w,subf_h]
		self.size = subf_w*subf_h
		face_size = face_w*face_h
		self.relative_size = self.size/face_size

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# PROCESSING PART FOR VIDEO STREAM
if not args.get('image'):
	while True:
		
		found_faces = []
		found_noses = []
		found_eyes = []
		found_mouths = []
		face_index = 0
		start = timer()

		(grabbed, frame) = camera.read()
		if args.get('video') and not grabbed:
			break
		frame = imutils.resize(frame, width = 300)

		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

		face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
		found_faces = face_cascade.detectMultiScale(gray)

		for (face_x,face_y,face_w,face_h) in found_faces:
			face_index += 1

			# face rectangles
			cv2.rectangle(frame,(face_x,face_y),(face_x+face_w,face_y+face_h),(230,223,208),2)

			# roi based on found_faces
			roi_gray = gray[face_y:face_y+face_h, face_x:face_x+face_w]
			roi_color = frame[face_y:face_y+face_h, face_x:face_x+face_w]

			# further cascades within roi
			nose_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml')
			eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
			mouth_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml')

			# applying cascades
			currently_found_noses = nose_cascade.detectMultiScale(roi_gray)
			currently_found_eyes = eye_cascade.detectMultiScale(roi_gray)
			currently_found_mouths = mouth_cascade.detectMultiScale(roi_gray)

			# append variable that's holding all the found subfeatures for each face

			# found_noses.append()

			#drawing rectangles
			for (ex,ey,ew,eh) in found_eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(53,30,247),1)
			for (nx,ny,nw,nh) in found_noses:
				cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(165,148,23),1)
			for (mx,my,mw,mh) in found_mouths:
				cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(48,32,3),1)

		cv2.imshow('processed_frame',frame)


	
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

		end = timer()
		print(end-start)
# ------------------------------------------------------------------------------
# PROCESSING PART FOR USER GIVEN IMAGE
else:

	result = []
	face_index = 0

	# img = imutils.resize(img, width = 300)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
	found_faces = face_cascade.detectMultiScale(gray)

	for (face_x,face_y,face_w,face_h) in found_faces:
		face_index += 1

		# face rectangles
		cv2.rectangle(img,(face_x,face_y),(face_x+face_w,face_y+face_h),(230,223,208),2)

		# roi based on found_faces
		roi_gray = gray[face_y:face_y+face_h, face_x:face_x+face_w]
		roi_color = img[face_y:face_y+face_h, face_x:face_x+face_w]

		# further cascades within roi
		nose_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml')
		eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
		mouth_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml')

			# applying cascades
		found_noses = nose_cascade.detectMultiScale(roi_gray)
		found_eyes = eye_cascade.detectMultiScale(roi_gray)
		found_mouths = mouth_cascade.detectMultiScale(roi_gray)

		# subfeature cascades
		cascades = []
		cascades.append(cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml'))
		cascades.append(cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml'))
		cascades.append(cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml'))

		# applying the above declared cascades
		found_subfeatures = []
		for cascade in cascades:
			found_subfeatures.append([])
			if len(cascade.detectMultiScale(roi_gray)) != 0:
				for element in cascade.detectMultiScale(roi_gray).tolist():
					found_subfeatures[-1].append(FoundSubFeature(face_index,element[0],element[1],element[2],element[3],face_x,face_y,face_w,face_h))

		result.append(found_subfeatures)


		#drawing rectangles
		for (ex,ey,ew,eh) in found_eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(53,30,247),1)
		for (nx,ny,nw,nh) in found_noses:
			cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(165,148,23),1)
		for (mx,my,mw,mh) in found_mouths:
			cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(48,32,3),1)

		# append variable that's holding all the found subfeatures for each face

		

	# logic to eliminate false positive results

	# found_noses = found_noses.tolist()

	# for i in range(len(found_noses)):
	# 	found_noses[i] = FoundObject(found_noses[i][0],found_noses[i][1],found_noses[i][2],found_noses[i][3])

	cv2.imshow('processed_image',img)
	cv2.moveWindow('processed_image',1000,300)
	cv2.waitKey(0)
		
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# cleanup time
if args.get('video'):
	camera.release()
cv2.destroyAllWindows()