import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesBasic'
#create list of image
images = []
personName = []
myList = os.listdir(path)       # to read the components of ImageBasic directory.
#print(myList)          #TEST
# to access the myList contant we use for loop..
for cu_img in myList:
    current_Img=cv2.imread(f"{path}/{cu_img}")
    images.append(current_Img)
    personName.append(os.path.splitext(cu_img)[0])      # get the only name from image extension like(abhi.jpg) // output abhi
#print(personName)

# create function for encoding the face
def faceEncoding(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]      # find out the encoding and give the 0th element
        encodeList.append(encode)
    return encodeList

#print(faceEncoding(images))            TEST
encodingListKnown = faceEncoding(images)
print("Successfully complete Encoding images!!!!")

# to read our cammera
cap=cv2.VideoCapture(0)     # if Laptop cammera put id 0 , otherwish put id 1

while True:
    ret,frame=cap.read()
    # resize the cammera..
    faces=cv2.resize(frame, (0,0), None, 0.25, 0.25)
    # change camera input video color BGR 2 RGB
    faces=cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    # find out the face location which value getting from camera
    facesCurrentFrame= face_recognition.face_locations(faces)

    # find out the face encoding which value getting from camera
    encodeCurrentFrame = face_recognition.face_encodings(faces,facesCurrentFrame)

# matching the face
    for encodeFace, faceLoc in zip(encodeCurrentFrame , facesCurrentFrame):
        matches=face_recognition.compare_faces(encodingListKnown, encodeFace)       # need to Compare the face
        faceDistance=face_recognition.face_distance(encodingListKnown,encodeFace)      # need to findout distance

        matchIndex=np.argmin(faceDistance)  # need to find the minimum distance

        if matches[matchIndex]:
            name=personName[matchIndex].upper()
            print(name)

            #create rectangle on face..
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)     # create rectangle with green color

    cv2.imshow("Camera", frame)         # to show image on camera
    if cv2.waitKey(1) == 13: # when we press ENTER buttom then close the window
        break
cap.release()
cv2.destroyAllWindows()















