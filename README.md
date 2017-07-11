
# <center>Object Detection using Haar feature-based cascade classifiers:<br><br>`Viola-Jones face detection`</center>
---
### <center> made for </center> ![](pictures/realeyes.png)
---

Within this document I will implement the so-called Viola-Jones face detector. Secondly I will implement a Haar feature-based cascade classifier for sub-face feature detection (e.g. eyes, nose or mouth detector). To improve this sub-feature detector, I will use biometric information, to filter false-positive results. To grade the classifier I have used two databases:<br>
* [IMM Frontal Face Database](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3943)
* [BioID Face Database](https://www.bioid.com/About/BioID-Face-Database)

<b>Databases are not included in this git repository. Please download these databases and place them in subfolder called databases. Proper folder names and folder structures are cruical for execution.</b>
***
This file is a `Jupyter notebook` using `python3` as a kernel. To execute, please install/run [jupyter](http://jupyter.org).<br>
Within this repository, there is a `violaJones.py` file as well. To evaluate the detector (improved detector also included) on live webcam video stream, execute that file without an argument. That `.py` file is also capable of detecting a single image with argument `-i`. (See usage examples at first rows of `violaJones.py`)
***
I have used the following external libraries (please, install those libraries if you want to execute the following cells):<br>
* [OpenCV](http://opencv.org)
* [numpy](http://www.numpy.org)
* [plotly](https://plot.ly)
* [imutils](https://github.com/jrosebr1/imutils)
---

### list the available haarcascades first

below with bold those, I have used through implementation

`ls /haarcascades/`
***
**`haarcascade_eye.xml`**

`haarcascade_eye_tree_eyeglasses.xml
haarcascade_frontalcatface.xml
haarcascade_frontalcatface_extended.xml
haarcascade_frontalface_alt.xml`

**`haarcascade_frontalface_alt2.xml`**

`haarcascade_frontalface_alt_tree.xml
haarcascade_frontalface_default.xml
haarcascade_fullbody.xml
haarcascade_lefteye_2splits.xml
haarcascade_licence_plate_rus_16stages.xml
haarcascade_lowerbody.xml
haarcascade_profileface.xml
haarcascade_righteye_2splits.xml
haarcascade_russian_plate_number.xml
haarcascade_smile.xml
haarcascade_upperbody.xml
haarcascade_mcs_eyepair_big.xml
haarcascade_mcs_eyepair_small.xml
haarcascade_mcs_leftear.xml
haarcascade_mcs_lefteye.xml
haarcascade_mcs_lefteye_alt.xml`

**`haarcascade_mcs_mouth.xml
haarcascade_mcs_nose.xml`**

`haarcascade_mcs_rightear.xml
haarcascade_mcs_righteye.xml
haarcascade_mcs_righteye_alt.xml
haarcascade_mcs_upperbody.xml`

***
### import necessary libraries:


```python
import numpy as np
import cv2
import argparse
import imutils
from timeit import default_timer as timer
import os
import time
import random
```

### `FoundSubFeature` Class holds advanced properties of subface-features instead of default `[x,y,w,h]` properties:


```python
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
        
```

## `detector` represents Viola-Jones detector:


```python
def detector(img,face_cascade,subfeature_cascades):

    results = []
    face_index = 0

    # img = imutils.resize(img, width = 300)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    found_faces = face_cascade.detectMultiScale(gray)

    for (face_x,face_y,face_w,face_h) in found_faces:
        
        face_index += 1

        # face rectangles
#         cv2.rectangle(img,(face_x,face_y),(face_x+face_w,face_y+face_h),(241,240,236),2)

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
```

### drawing function based on `FoundSubFeature` Class:


```python
def drawRectangleForSubFeature(img,FoundSubFeature,color):   
    vertex_1 = tuple([int(x) for x in [sum(element) for element in zip(FoundSubFeature.face_midpoint,FoundSubFeature.relative_midpoint,[x / -2 for x in FoundSubFeature.bounding_box])]])
    vertex_2 = tuple([int(x) for x in [sum(element) for element in zip(FoundSubFeature.face_midpoint,FoundSubFeature.relative_midpoint,[x / 2 for x in FoundSubFeature.bounding_box])]])
    cv2.rectangle(img,vertex_1,vertex_2,color,1)

def drawImageDetection(image,result,features):
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
```

### proper inline image display in `Jupyter notebook` requires a custom function:


```python
from matplotlib import pyplot as plt
def showImage(image):
    b,g,r = cv2.split(image)
    img_rgb = cv2.merge([r,g,b])
    
    figure = plt.figure(figsize = (100,10))
    plt.imshow(img_rgb)
    plt.show()
```

### Class to hold the image itself and its detected result:


```python
class DetectedImage:
    def __init__(self,image,result):
        self.image = image
        self.result = result
```

---
# evaluate Viola-Jones detector (without parameters) for each images in user defined `directory`*
### *these are unlabelled pictures


```python
%%time
# Wall time: ~ 53.8s | CPU times: ~ 2min 27s

directory = './databases/IMM-Frontal-Face-DB-SMALL' # easy database
# directory = './databases/vision.caltech.edu/' # advanced database (was not used)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
cascades = []
cascades.append(cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml'))
cascades.append(cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml'))
cascades.append(cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml'))

features = ['eye','nose','mouth'] # used for printing on image

detected_images = []
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        filepath = os.path.join(directory,filename)
        
        image = cv2.imread(filepath)
        result = detector(image,face_cascade=face_cascade,subfeature_cascades=cascades)
        
        detected_images.append(DetectedImage(image,result))
```

    CPU times: user 2min 27s, sys: 1.12 s, total: 2min 28s
    Wall time: 57 s


## display result


```python
imageindex = random.randint(0,len(detected_images))
showImage(drawImageDetection(detected_images[imageindex].image,detected_images[imageindex].result,features))
print('PICTURE ID.:'+str(imageindex))
print('printing relative_midpoint_percentage and relative_size_percentage for each detected rectangles:')
print('-------------------- EYES -------------------------')
for eye in detected_images[imageindex].result[0][0]:
    print(eye.relative_midpoint_percentage,eye.relative_size_percentage)
print('-------------------- NOSE -------------------------')
for nose in detected_images[imageindex].result[0][1]:
    print(nose.relative_midpoint_percentage,nose.relative_size_percentage)
print('-------------------- MOUTH -------------------------')
for mouth in detected_images[imageindex].result[0][2]:
    print(mouth.relative_midpoint_percentage,mouth.relative_size_percentage)
```


![png](output_18_0.png)


    PICTURE ID.:32
    printing relative_midpoint_percentage and relative_size_percentage for each detected rectangles:
    -------------------- EYES -------------------------
    [-21.201413427561839, -11.66077738515901] 4.05673688022
    [18.551236749116608, -10.777385159010601] 4.20032713606
    [4.4169611307420498, 15.01766784452297] 0.60432768545
    [-7.4204946996466434, 13.074204946996467] 1.05008178402
    -------------------- NOSE -------------------------
    [-0.35335689045936397, -20.671378091872793] 34.7313613605
    [-1.5901060070671376, 9.8939929328621901] 4.83961592728
    [-13.780918727915195, 27.031802120141339] 3.15898562849
    -------------------- MOUTH -------------------------
    [20.141342756183743, -8.6572438162544181] 6.50276567319
    [-19.257950530035338, -8.8339222614840995] 7.21946834147
    [0.0, -3.1802120141342751] 34.6302238759
    [-3.5335689045936398, 34.628975265017672] 6.24929765636
    [-0.88339222614840995, 16.431095406360424] 6.06824907291


## first eval remarks:
### _positive<br>all the keypoints were found
### _negative<br>without parameterisation or post-detection filtering, there are too many false-positive results. 

---
# <center>Grade `Viola-Jones` classifier</center>
#### <center>based on</center>
## <center>[BioID Face Database - FaceDB](https://www.bioid.com/About/BioID-Face-Database)</center> 
---
### <center>consists of 1521 labelled images, 20 keypoints each</center>
![](pictures/bioid_keypoints.gif)

`Legend:
There are 20 manually placed points on each 1521 images.
The markup scheme is as follows:
0 = right eye pupil
1 = left eye pupil
2 = right mouth corner
3 = left mouth corner
4 = outer end of right eyebrow
5 = inner end of right eyebrow
6 = inner end of left eyebrow
7 = outer end of left eyebrow
8 = right temple
9 = outer corner of right eye
10 = inner corner of right eye
11 = inner corner of left eye
12 = outer corner of left eye
13 = left temple
14 = tip of nose
15 = right nostril
16 = left nostril
17 = centre point on outer edge of upper lip
18 = centre point on outer edge of lower lip
19 = tip of chin`

### create a  class, holding above mentioned keypoints


```python
class LabelledPicture:
    def __init__(self,picture,right_eye_pupil, left_eye_pupil, right_mouth_corner, left_mouth_corner, outer_end_of_right_eye_brow, inner_end_of_right_eye_brow, inner_end_of_left_eye_brow, outer_end_of_left_eye_brow, right_temple, outer_corner_of_right_eye, inner_corner_of_right_eye, inner_corner_of_left_eye, outer_corner_of_left_eye, left_temple, tip_of_nose, right_nostril, left_nostril, centre_point_on_outer_edge_of_upper_lip, centre_point_on_outer_edge_of_lower_lip, tip_of_chin):
        self.picture = picture
        self.right_eye_pupil = right_eye_pupil
        self.left_eye_pupil = left_eye_pupil
        self.right_mouth_corner = right_mouth_corner
        self.left_mouth_corner = left_mouth_corner
        self.outer_end_of_right_eye_brow = outer_end_of_right_eye_brow
        self.inner_end_of_right_eye_brow = inner_end_of_right_eye_brow
        self.inner_end_of_left_eye_brow = inner_end_of_left_eye_brow
        self.outer_end_of_left_eye_brow = outer_end_of_left_eye_brow
        self.right_temple = right_temple
        self.outer_corner_of_right_eye = outer_corner_of_right_eye
        self.inner_corner_of_right_eye = inner_corner_of_right_eye
        self.inner_corner_of_left_eye = inner_corner_of_left_eye
        self.outer_corner_of_left_eye = outer_corner_of_left_eye
        self.left_temple = left_temple
        self.tip_of_nose = tip_of_nose
        self.right_nostril = right_nostril
        self.left_nostril = left_nostril
        self.centre_point_on_outer_edge_of_upper_lip = centre_point_on_outer_edge_of_upper_lip
        self.centre_point_on_outer_edge_of_lower_lip = centre_point_on_outer_edge_of_lower_lip
        self.tip_of_chin = tip_of_chin
        
        self.mouth_midpoint = [(self.left_mouth_corner[0]+self.right_mouth_corner[0])/2,(self.left_mouth_corner[1]+self.right_mouth_corner[1])/2]
```

### read BioID Face Database


```python
labelled_pictures = []
directory = './databases/BioID-FaceDatabase-V1/pictures/'

for filename in os.listdir(directory):
    if filename.endswith('.pgm'):
        filepath = os.path.join(directory,filename)
        
        picture = cv2.imread(filepath)
        points_filepath = os.path.join(directory,'points/'+filename[:-4]+'.pts')
        
        points_file = open(points_filepath,'r')
        points_file.seek(29)
        variable_number = -1
        variables = []
        for line in points_file:
            variable_number += 1
            if variable_number <20:
                separator_position = line.find(' ')
                end_position = line.find('\r')
                x = float(line[0:separator_position])
                y = float(line[separator_position+1:end_position])
                variables.append([x,y])
        labelled_pictures.append(LabelledPicture(picture,variables[0], variables[1], variables[2], variables[3], variables[4], variables[5], variables[6], variables[7], variables[8], variables[9], variables[10], variables[11], variables[12], variables[13], variables[14], variables[15], variables[16], variables[17], variables[18], variables[19]))
```

### evaluate `Viola-Jones` classifier on BioID Database


```python
%%time
# Wall time: ~ 3 min | CPU times: ~ 8min 20s

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
cascades = []
cascades.append(cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml'))
cascades.append(cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml'))
cascades.append(cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml'))

features = ['eye','nose','mouth'] # used for printing on image

detected_images = []
for element in labelled_pictures:
#     filepath = os.path.join(directory,filename)

    image = element.picture
    if 'improvedDetector' in locals():
        result = improvedDetector(image,face_cascade,cascades,intervals)
    else:
        result = detector(image,face_cascade,cascades)

    detected_images.append(DetectedImage(image,result))
```

    CPU times: user 8min 24s, sys: 1.97 s, total: 8min 26s
    Wall time: 3min 3s


**`labelled_pictures`**`: BioID labelled database`<br>
**`detected_images`**`: Viola-Jones detector's result`

### relative_distance counts distance between labelled keypoint and detected keypoint


```python
def relative_distance(labelled_position,detected_position,image_shape):
    absolute_distance = ((labelled_position[0]-detected_position[0])**2+(labelled_position[1]-detected_position[1])**2)**0.5
    image_diagonal = (image_shape[0]**2+image_shape[1]**2)**0.5
    relative_distance = absolute_distance / image_diagonal * 100
    return relative_distance
```

### Grade holds grade for each `Viola-Jones` detected keypoints
##### grading system will be built on these Class properties


```python
class Grade:
    def __init__(self,face_count,lefteye_found,lefteye_redundant,lefteye_falsepositive_counter,righteye_found,righteye_redundant,righteye_falsepositive_counter,nose_found,nose_redundant,nose_falsepositive_counter,mouth_found,mouth_redundant,mouth_falsepositive_counter):

        self.face_count = face_count
        
        self.lefteye_found = lefteye_found
        self.lefteye_redundant = lefteye_redundant
        self.lefteye_falsepositive_counter = lefteye_falsepositive_counter
        
        self.righteye_found = righteye_found
        self.righteye_redundant = righteye_redundant
        self.righteye_falsepositive_counter = righteye_falsepositive_counter
        
        self.nose_found = nose_found
        self.nose_redundant = nose_redundant
        self.nose_falsepositive_counter = nose_falsepositive_counter
        
        self.mouth_found = mouth_found
        self.mouth_redundant = mouth_redundant
        self.mouth_falsepositive_counter = mouth_falsepositive_counter
```

### grades is list of Grade – for all the images within dataset


```python
tolerance = 5 # percentage

grades = []

for image_index in range(len(detected_images)):
    
    grade = Grade(0,False,0,0,False,0,0,False,0,0,False,0,0)
    
    image_shape = labelled_pictures[image_index].picture.shape[:2]
    
    labelled_mouth_position = labelled_pictures[image_index].mouth_midpoint
    labelled_nose_position = labelled_pictures[image_index].tip_of_nose
    labelled_lefteye_position = labelled_pictures[image_index].left_eye_pupil
    labelled_righteye_position = labelled_pictures[image_index].right_eye_pupil

    faces = detected_images[image_index].result
    grade.face_count = len(faces)
        
    for face in faces:
        [eyes, noses, mouths] = face
        for eye in eyes:
            if relative_distance(eye.midpoint_global,labelled_lefteye_position,image_shape) < tolerance:
                if grade.lefteye_found == True:
                    grade.lefteye_redundant += 1
                grade.lefteye_found = True
            elif relative_distance(eye.midpoint_global,labelled_righteye_position,image_shape) < tolerance:
                if grade.righteye_found == True:
                    grade.righteye_redundant += 1
                grade.righteye_found = True
            else:
                grade.lefteye_falsepositive_counter += 1
                grade.righteye_falsepositive_counter += 1
        for nose in noses:
            if relative_distance(nose.midpoint_global,labelled_nose_position,image_shape) < tolerance:
                if grade.nose_found == True:
                    grade.nose_redundant += 1
                grade.nose_found = True
            else:
                grade.nose_falsepositive_counter += 1
        for mouth in mouths:
            if relative_distance(mouth.midpoint_global,labelled_mouth_position,image_shape) < tolerance:
                if grade.mouth_found == True:
                    grade.mouth_redundant += 1
                grade.mouth_found = True
            else:
                grade.mouth_falsepositive_counter += 1
    grade.cumulated_found = grade.lefteye_found + grade.righteye_found + grade.nose_found + grade.mouth_found
    grade.cumulated_falsepositive = grade.lefteye_falsepositive_counter + grade.righteye_falsepositive_counter + grade.nose_falsepositive_counter + grade.mouth_falsepositive_counter
    grade.cumulated_redundancy = grade.lefteye_redundant + grade.righteye_redundant + grade.nose_redundant + grade.mouth_redundant
    grades.append(grade)
    
```


```python
# showImage(image=drawImageDetection(image=detected_images[15].image,result=detected_images[15].result,features=['eye','nose','mouth']))
```


```python
for detection in grades:
    if detection.cumulated_found == 4: # means all the subfeatures were found
        if detection.cumulated_falsepositive == 0:
            if detection.cumulated_redundancy == 0:
                score = '6'
            else:
                score = '5'
        else:
            score = '4'
    else:
        score = str(detection.cumulated_found)
    if detection.face_count      != 0 and \
       detection.cumulated_found == 0 :
        score = 'r'
        
    if detection.face_count == 1:
        if score != 'r':
            score += '+'
    elif detection.face_count > 1:
        if score != 'r':
            score += '-'
    detection.score = score
```


```python
score_count = {'6+':0,'6-':0,'5+':0,'5-':0,'4+':0,'4-':0,'3+':0,'3-':0,'2+':0,'2-':0,'1+':0,'1-':0,'0':0,'r':0,}

for detection in grades:
    score_count[detection.score] += 1
```

# description of scores

|`score`|                              `detail`                             |`score_count(+,-)`|`rgba_color`     |
|-------| -------------------------------------------------------------------- |-------------|-----------------|
| **6** | all features were found w/o redundancy w/o false_positive detection  |                                         {{score_count['6+'],score_count['6-']}}                                                |rgba(14,173,0,#) |
| **5** | all features were found w/  redundancy w/o false_positive detection  |                                         {{score_count['5+'],score_count['5-']}}                                                |rgba(140,198,0,#)|
| **4** | all features were found w/  redundancy w/  false_positive detection  |                                         {{score_count['4+'],score_count['4-']}}                                                |rgba(150,169,0,#)|
| **3** | 1 feature  was  not found                                            |                                         {{score_count['3+'],score_count['3-']}}                                                |rgba(254,251,2,#)|
| **2** | 2 features were not found                                            |                                         {{score_count['2+'],score_count['2-']}}                                                |rgba(254,196,0,#)|
| **1** | 3 features were not found                                            |                                         {{score_count['1+'],score_count['1-']}}                                                |rgba(255,148,0,#)|
|       |                                                                      |             |                 |
| **+** | only one face was found                                              |             |rgba(#,#,#,1)    |
| **-** | more than one face were found                                        |             |rgba(#,#,#,0.5)  |
|       |                                                                      |             |                 |
| **0** | no faces (therefore no subfeatures) were found                       |                                         {{score_count['0']}}                                                                   |rgba(254,37,0,1) |
| **r** | (r = review) no features were found but a face                       |                                         {{score_count['r']}}                                                                   |rgba(1,162,199,1)|


```python
divisor = sum(list(score_count.values()))

distribution = [ (score_count['6+']+score_count['6-']) / divisor * 100,
                 (score_count['5+']+score_count['5-']) / divisor * 100,
                 (score_count['4+']+score_count['4-']) / divisor * 100,
                 (score_count['3+']+score_count['3-']) / divisor * 100,
                 (score_count['2+']+score_count['2-']) / divisor * 100,
                 (score_count['1+']+score_count['1-']) / divisor * 100,
                 score_count['0'] / divisor * 100,
                 score_count['r'] / divisor * 100
               ]
```

|                              `detail`                              |percentage|
| ------------------------------------------------------------------ ||
| all features were found w/o redundancy w/o false_positive detection|{{distribution[0]}}|
| all features were found w/  redundancy w/o false_positive detection|{{distribution[1]}}|
| all features were found w/  redundancy w/  false_positive detection|{{distribution[2]}}|
| 1 feature  was  not found                                          |{{distribution[3]}}|
| 2 features were not found                                          |{{distribution[4]}}|
| 3 features were not found                                          |{{distribution[5]}}|
| no faces (therefore no subfeatures) were found                     |{{distribution[6]}}|
| no features were found but a face                                  |{{distribution[7]}}|


```python
colors = ['rgba(14,173,0,1)' ,'rgba(14,173,0,0.65)' ,\
          'rgba(140,198,0,1)','rgba(140,198,0,0.65)',\
          'rgba(150,169,0,1)','rgba(150,169,0,0.65)',\
          'rgba(254,251,2,1)','rgba(254,251,2,0.65)',\
          'rgba(254,196,0,1)','rgba(254,196,0,0.65)',\
          'rgba(255,148,0,1)','rgba(255,148,0,0.65)',\
          'rgba(254,37,0,1)' ,'rgba(1,162,199,1)']
```


```python
import plotly
from plotly.graph_objs import *

plotly.tools.set_credentials_file(username='makejunk', api_key='RP2u7YG4QuOZe8FcBXU3')
plotly.offline.init_notebook_mode(connected=True)

fig = {
    'data': [
        {
            'values': list(score_count.values()),
            'type': 'pie',
            'labels': list(score_count.keys()),
            'marker': {'colors': colors,
                       'line': dict(color='#FFFFFF',width=1)
                      },
            'textinfo':'percent',
            'hoverinfo' : 'label'
        }   ],
    'layout': {
        'title': 'Distribution of scores',
        'showlegend': True,
              }
      }

plotly.plotly.iplot(fig)
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>





<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~makejunk/205.embed" height="525px" width="100%"></iframe>




```python
data = plotly.graph_objs.Bar(
    x=list(score_count.keys()),
    y=list(score_count.values()),
    marker=dict(
        color=colors)
)

layout = plotly.graph_objs.Layout(
    title='Distribution of scores',
)

fig = plotly.graph_objs.Figure(data=[data], layout=layout)
plotly.plotly.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~makejunk/207.embed" height="525px" width="100%"></iframe>



___
# improve Viola-Jones, based on sub-feature positions and sizes
#### based on the following self labelled samples of IMM-Frontal-Face Database
![](pictures/self_labelled_sample.png)


```python
class Intervals:
    def __init__(self,lefteye_relative_midpoint_percentage,righteye_relative_midpoint_percentage,eye_relative_size_percentage,nose_relative_midpoint_percentage,nose_relative_size_percentage,mouth_relative_midpoint_percentage,mouth_relative_size_percentage):
        self.lefteye_relative_midpoint_percentage = lefteye_relative_midpoint_percentage
        self.righteye_relative_midpoint_percentage = righteye_relative_midpoint_percentage
        self.eye_relative_size_percentage = eye_relative_size_percentage
        self.nose_relative_midpoint_percentage = nose_relative_midpoint_percentage
        self.nose_relative_size_percentage = nose_relative_size_percentage
        self.mouth_relative_midpoint_percentage = mouth_relative_midpoint_percentage
        self.mouth_relative_size_percentage = mouth_relative_size_percentage
```


```python
intervals = Intervals(
    lefteye_relative_midpoint_percentage =  [ [10,30] , [-20,-5] ],\
    righteye_relative_midpoint_percentage = [ [-30,-10] , [-20,-5] ],\
    eye_relative_size_percentage =          [0,20],\
    nose_relative_midpoint_percentage =     [ [-5,5] , [5,20] ],\
    nose_relative_size_percentage =         [2,15],\
    mouth_relative_midpoint_percentage =    [ [-4,4] , [25,45] ],\
    mouth_relative_size_percentage =        [3,13]
                    )
```


```python
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
```

## To evaluate detection go back to cell evaluate `Viola-Jones` classifier on BioID Database and execute all the cells once again
##### This time, improvedDetector() will be used, insted of detector() ... please take a look at the diagrams and note the improvements

# Compare detector() and improvedDetector()
![](pictures/photo_sample.png)
![](pictures/video_sample.png)

# Closing remarks

All the above implemented detectors, included logic, test results could be examined quite deep. So far, I could implement the above written methods. Before ending, let me add a few closing notes.

## Improvement notes

* On current databases, improvedDetector() seems to do a much better job, but on e.g. slightly tilted faces they might decrease correct detection rate.

## What's missing?

* Include more datasets.
* Test on advanced-level datasets.
* Visualise improvements better: implement statistical analysis.
* Sophisticated statistical analysis of both false-positive and false-negative mistakes, and also biometric parameters (presuming normal distribution, many suggestions for above mentioned intervals could be improved).

## Other methods

* Viola-Jones with custom trained haarcascades.
* Convolutional Neural Networks.
* HOG / SVM.
