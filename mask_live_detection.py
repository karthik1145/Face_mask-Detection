from datetime import datetime
import geocoder
import numpy as np



from mtcnn import MTCNN
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from resizeimage import resizeimage

detector=MTCNN()
model=load_model('/Users/pulla/Downloads/face_mask3.h5')

font1=cv2.FONT_HERSHEY_TRIPLEX
font2=cv2.FONT_HERSHEY_SIMPLEX

cap=cv2.VideoCapture(0)

while(True):
      _,frame=cap.read()
      #cv2.putText(frame,str(datetime.now()),(10,30),font1,1,(255,255,255),2,cv2.LINE_AA)
      #cv2.putText(frame,Loc,(10,450),font2,1,(255,255,255),2,cv2.LINE_AA)
      #cv2.putText(frame,'Pk',(480,450),font1,1,(255,255,255),2,cv2.LINE_AA)
      result=detector.detect_faces(frame)

      for person in result:
          bounding_box=person['box']
          keypoints=person['keypoints']
          cv2.imwrite('opencv.png',frame)
          image_file=load_img('opencv.png',target_size=(160,160))

          x=img_to_array(image_file)
          x=np.expand_dims(x,axis=0)
          pred=model.predict(x)
          if(pred[0][0]==0.0):
            cv2.putText(frame,"ACCESS GRANTED, MASK ON",(100,100),font1,0.8,(0,255,0),2,cv2.LINE_AA)
          elif pred[0][0]==1.0:
            cv2.putText(frame,"ACCESS Denied,No MASK! ",(100,100),font1,0.8,(0,0,255),2,cv2.LINE_AA)

          cv2.rectangle(
              frame,
              (bounding_box[0],bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2
          )
          cv2.circle(frame,(keypoints['left_eye']),2,(0,155,255),2)
          cv2.circle(frame,(keypoints['right_eye']),2,(0,155,255),2)
          cv2.circle(frame,(keypoints['nose']),2,(0,155,255),2)
          cv2.circle(frame,(keypoints['mouth_left']),2,(0,155,255),2)
          cv2.circle(frame,(keypoints['mouth_right']),2,(0,155,255),2)


      cv2.imshow('frame',frame)

      if cv2.waitKey(5) &0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

