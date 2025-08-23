import cv2 as cv 
# import psycopg2 as db
# import pyautogui
import mediapipe  as mp
import time


class Facerecognition:
    'came for camera used 0 for pc cam and 1 for  external cam'
    'cascade for recognition of face or eye or object or more'
    def __init__(self,came = 0,cascade_path = "haarcascade_frontalface_default.xml"):
        self.video = cv.VideoCapture(came)
        self.cascade_path = cv.CascadeClassifier(cv.data.haarcascades + cascade_path) # type:ignore
        self.hands =mp.solutions.hands
        self.utils=mp.solutions.drawing_utils
        self.Hands = self.hands.Hands()


        self.pictures = []

    def video_capture_gray(self,file_name,show = False):
        fourcc = cv.VideoWriter_fourcc(*"XVID") # type: ignore
        width =  int(self.video.get(cv.CAP_PROP_FRAME_WIDTH))
        height =  int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT))
        file = cv.VideoWriter(file_name,fourcc,25,(width,height),isColor=False)


        while True:
             ret,frame = self.video.read()
             if ret:
                 gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

                 #saving gray file                  
                 file.write(gray)

                 if show is True:
                     cv.imshow("gray_image",gray)
                    
                     if cv.waitKey(1) & 0xFF == ord("q"):
                         break
                
        self.video.release()
        file.release()
        cv.destroyAllWindows()
                
    def video_show_face(self):
         while True:
             ret,frame = self.video.read()
              
             if ret:
                 gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
                 self.pictures.append(gray)

                 face = self.cascade_path.detectMultiScale(gray,1.1,5,minSize=(20,20))


                 for (x,y,w,h) in face:
                     cv.rectangle(frame,(x,y-10),(x+w,y+h),(0,255,0),2,)
                    
                     '''frame is the array img , then the cordinates that will be show in the rectangle box
                        font : style
                        font scale : how the font will apper in size 0.5 is small and 2 is bigger 
                        color : any
                        pix  : smaller recommonded
                     '''
                     cv.putText(frame,"face detected",(x,y -20),cv.FONT_HERSHEY_SIMPLEX,0.82,(255,0,0),2,cv.LINE_AA)
                    
                
                 cv.imshow("image",frame)

                 if cv.waitKey(1) & 0xFF == ord("q"):
                         break
             else:
                 print('cannot capture')
                 break
                 
         self.video.release()
         cv.destroyAllWindows()

    def pictures_(self):
        print(self.pictures)

    def hand_landmark(self):
        ctime = 0 
        ptime = 0
        while True:
             ret,frame = self.video.read()
              
             if ret:
                 gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
                 rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    
                #  hands= self.hands.Hands(
                #  static_image_mode=False,      # True for images, False for video
                #    max_num_hands=2,              # Max hands to detect
                #    min_detection_confidence=0.7, # Min confidence to detect hands
                #    min_tracking_confidence=0.5   # Min confidence to track hands
                #  )
                 result = self.Hands.process(rgb)
                 if result.multi_hand_landmarks:
                      for handmarks in result.multi_hand_landmarks:
                           self.utils.draw_landmarks(frame,handmarks, self.hands.HAND_CONNECTIONS, self.utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1), self.utils.DrawingSpec(color=(0,255,0), thickness=2))
        
                           

                 ctime = time.time()
                 fps = 1/(ctime-ptime) if (ctime - ptime) > 0 else 0
                 # when ever the frame change it will update thevalue for calcutation
                 ptime = ctime
                 cv.putText(frame,f"fps{int(fps)}",(10,20),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0))

                 cv.imshow("Pointers",frame)



             if cv.waitKey(1) & 0xFF == ord("q"):
                  break
        self.video.release()
        cv.destroyAllWindows() 

    
    



if __name__ == "__main__":
        
    # camera = input("0 or 1 :")
    # cascade = input("Enter cascade xml :")
    # fr = Facerecognition(camera,cascade) #"haarcascade_frontalface_default.xml"
    # # fr.video_show_face()        
    # # fr.pictures_()
    # fr.video_capture_gray("image.mp4",True)


        #  "while True:""""
        #  print(pyautogui.position()) # x = 1775 , y = 64"""
        #  x= 1775
        #  y = 64
        # #  x,y = pyautogui.position()
        fr = Facerecognition()
        fr.hand_landmark()