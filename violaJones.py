# USAGE1: python3 violaJones.py                          ->  webcamera will be used for video stream
# USAGE2: python3 violaJones.py -i user_give_image_path  ->  detect on single image

# ------------------------------------------------------------------------------

import numpy as np
import cv2
import argparse
import imutils
from timeit import default_timer as timer
import copy

# ------------------------------------------------------------------------------


ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='path to the (optional) video file')
ap.add_argument('-i', '--image', help='path to the (optional) image file')
args = vars(ap.parse_args())

# ------------------------------------------------------------------------------


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

class FoundSubFeature:
	def __init__(self,parent,subf_x,subf_y,subf_w,subf_h,face_x,face_y,face_w,face_h):
		self.parent = parent
		self.midpoint = [subf_x+subf_w/2,subf_y+subf_h/2]
		self.midpoint_global = [self.midpoint[0]+face_x,self.midpoint[1]+face_y]

		self.face_midpoint = [face_x+face_w/2,face_y+face_h/2]
		self.relative_midpoint = [self.midpoint[0]-face_w/2,self.midpoint[1]-face_h/2]
		self.relative_midpoint_percentage = [self.relative_midpoint[0]/face_w*100,self.relative_midpoint[1]/face_h*100]

		self.bounding_box = [subf_w,subf_h]
		self.size = subf_w*subf_h
		self.face_box = [face_w,face_h]
		face_size = face_w*face_h
		self.relative_size_percentage = self.size/face_size*100

		self.face_w = face_w
		
# ------------------------------------------------------------------------------

def detector(img,face_cascade,subfeature_cascades):

	results = []
	face_index = 0

	# img = imutils.resize(img, width = 300)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	found_faces = face_cascade.detectMultiScale(gray)

	for (face_x,face_y,face_w,face_h) in found_faces:
		
		face_index += 1

		# face rectangles
		cv2.rectangle(img,(face_x,face_y),(face_x+face_w,face_y+face_h),(241,240,236),2)

		# roi based on found_faces
		roi_gray = gray[face_y:face_y+face_h, face_x:face_x+face_w]
		roi_color = img[face_y:face_y+face_h, face_x:face_x+face_w]

		# applying subfeature_cascades
		found_subfeatures = []
		
		for cascade in subfeature_cascades:
			found_subfeatures.append([])
			objects = cascade.detectMultiScale(roi_gray,scaleFactor=1.05)
			if len(objects) != 0:
				for element in objects.tolist():
					found_subfeatures[-1].append(FoundSubFeature(face_index,element[0],element[1],element[2],element[3],face_x,face_y,face_w,face_h))
		results.append(found_subfeatures)
		
	return results

# ------------------------------------------------------------------------------

def drawImageDetection(image,result,features):

	def drawRectangleForSubFeature(img,FoundSubFeature,color):   
		vertex_1 = tuple([int(x) for x in [sum(element) for element in zip(FoundSubFeature.face_midpoint,FoundSubFeature.relative_midpoint,[x / -2 for x in FoundSubFeature.bounding_box])]])
		vertex_2 = tuple([int(x) for x in [sum(element) for element in zip(FoundSubFeature.face_midpoint,FoundSubFeature.relative_midpoint,[x / 2 for x in FoundSubFeature.bounding_box])]])
		cv2.rectangle(img,vertex_1,vertex_2,color,1)

	clouds = [241, 240, 236]
	alizarin = [60, 76, 231]
	peterriver = [219, 152, 52]
	nephritis = [96, 174, 39]
	colors = [clouds,alizarin,peterriver,nephritis]
	# len(colors) must not be less than len(features)
	
	for face in result:
		for subfeature in face:
			for element in subfeature:
				drawRectangleForSubFeature(image,element,colors[face.index(subfeature)])
	
	image_height = image.shape[0]
	image_width = image.shape[1]
	for feature in features:
		(x,y) = (int(image_width*0.7),int(image_height*0.3+features.index(feature)*50))
		cv2.putText(image,feature,(x,y),cv2.FONT_HERSHEY_SIMPLEX, int(image_width*0.005), colors[features.index(feature)])
	return image

# ------------------------------------------------------------------------------

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
cascades = []
cascades.append(cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml'))
cascades.append(cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml'))
cascades.append(cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml'))

# ------------------------------------------------------------------------------

class Intervals:
	def __init__(self,lefteye_relative_midpoint_percentage,righteye_relative_midpoint_percentage,eye_relative_size_percentage,nose_relative_midpoint_percentage,nose_relative_size_percentage,mouth_relative_midpoint_percentage,mouth_relative_size_percentage):
		self.lefteye_relative_midpoint_percentage = lefteye_relative_midpoint_percentage
		self.righteye_relative_midpoint_percentage = righteye_relative_midpoint_percentage
		self.eye_relative_size_percentage = eye_relative_size_percentage
		self.nose_relative_midpoint_percentage = nose_relative_midpoint_percentage
		self.nose_relative_size_percentage = nose_relative_size_percentage
		self.mouth_relative_midpoint_percentage = mouth_relative_midpoint_percentage
		self.mouth_relative_size_percentage = mouth_relative_size_percentage

# ------------------------------------------------------------------------------

intervals = Intervals(
	lefteye_relative_midpoint_percentage =  [ [ 10, 30] , [-20,-5] ],\
	righteye_relative_midpoint_percentage = [ [-30,-10] , [-20,-5] ],\
	eye_relative_size_percentage =		  	[0,20]                  ,\
	nose_relative_midpoint_percentage =	 	[ [ -5,  5] , [  5,20] ],\
	nose_relative_size_percentage =		 	[2,15]                  ,\
	mouth_relative_midpoint_percentage =	[ [ -4,  4] , [ 25,45] ],\
	mouth_relative_size_percentage =		[3,13]
					)

# ------------------------------------------------------------------------------

def improvedDetector(img,face_cascade,subfeature_cascades,intervals):
	temporary_results = detector(img,face_cascade,subfeature_cascades)
	
	def isInInterval(number,interval):
		if interval[0] <= number <= interval[1]:
			return True
		else:
			return False
	
	improved_results = []
	
	for face in temporary_results:
		improved_results.append([[],[],[]]) # empty face created
		[eyes, noses, mouths] = face
		for eye in eyes:
			if (isInInterval(eye.relative_midpoint_percentage[0],intervals.lefteye_relative_midpoint_percentage[0]) and isInInterval(eye.relative_midpoint_percentage[1],intervals.lefteye_relative_midpoint_percentage[1]))\
			or (isInInterval(eye.relative_midpoint_percentage[0],intervals.righteye_relative_midpoint_percentage[0]) and isInInterval(eye.relative_midpoint_percentage[1],intervals.righteye_relative_midpoint_percentage[1])):
				improved_results[-1][0].append(eye)
		for nose in noses:
			if isInInterval(nose.relative_midpoint_percentage[0],intervals.nose_relative_midpoint_percentage[0]) and isInInterval(nose.relative_midpoint_percentage[1],intervals.nose_relative_midpoint_percentage[1]):
				improved_results[-1][1].append(nose)
		for mouth in mouths:
			if isInInterval(mouth.relative_midpoint_percentage[0],intervals.mouth_relative_midpoint_percentage[0]) and isInInterval(mouth.relative_midpoint_percentage[1],intervals.mouth_relative_midpoint_percentage[1]):
				improved_results[-1][2].append(mouth)
				
	return improved_results

# ------------------------------------------------------------------------------

# PROCESSING PART FOR VIDEO STREAM
if not args.get('image'):
	while True:

		start = timer()

		(grabbed, frame) = camera.read()
		if args.get('video') and not grabbed:
			break
		frame = imutils.resize(frame, width = 500)
		improved_frame = copy.copy(frame)

		results = detector(frame,face_cascade,cascades)
		improved_results = improvedDetector(improved_frame,face_cascade,cascades,intervals)

		drawImageDetection(frame,results,['eye','nose','mouth'])
		drawImageDetection(improved_frame,improved_results,['eye','nose','mouth'])

		cv2.imshow('detection',frame)
		cv2.imshow('improved detection',improved_frame)


	
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

		end = timer()
		print(end-start)

# ------------------------------------------------------------------------------
# PROCESSING PART FOR USER GIVEN IMAGE
else:

	img = imutils.resize(img, width = 300)
	improved_img = copy.copy(img)

	results = detector(img,face_cascade,cascades)
	improved_results = improvedDetector(improved_img,face_cascade,cascades,intervals)

	drawImageDetection(img,results,['eye','nose','mouth'])
	drawImageDetection(improved_img,improved_results,['eye','nose','mouth'])

	cv2.imshow('detection',img)
	cv2.imshow('improved detection',improved_img)
	cv2.waitKey(0)
		
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# cleanup time
if args.get('video'):
	camera.release()
cv2.destroyAllWindows()