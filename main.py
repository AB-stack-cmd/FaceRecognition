import cv2 as cv

class Facerecognition:
    'came for camera used 0 for pc cam and 1 for  external cam'
    'cascade for recognition of face or eye or object or more'
    def __init__(self,came,cascade_path):
        self.video = cv.VideoCapture(came)
        self.cascade_path = cv.CascadeClassifier(cv.data.haarcascades + cascade_path)

        pictures = []

    def video_capture_gray(self,file_name,show = False):
        while True:
             ret,frame = self.video.read()
             if ret:
                 gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

                 file = cv.imwrite(file_name,gray)
                 print("captured gray image")

                 if show is True:
                     cv.imshow("gray_image",gray)
                    
                     if cv.waitKey(1) & 0xFF == ord("q"):
                         break
                 else: 
                     break
        self.video.release()
        cv.destroyAllWindows()
                
    def video_show_face(self):
         while True:
             ret,frame = self.video.read()
             if ret:
                 gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

                 face = self.cascade_path.detectMultiScale(gray,1.1,5,minSize=(20,20))


                 for (x,y,w,h) in face:
                     cv.rectangle(frame,(x,y-10),(x+w,y+h),(0,255,0),2,)
                    
                     '''frame is the array img , then the cordinates that will be show in the rectangle box
                        font : style
                        font scale : how the font will apper in size 0.5 is small and 2 is bigger 
                        color : any
                        pix  : smaller recommonded
                     '''
                     cv.putText(frame,"face detected",(x,y -20),cv.FONT_HERSHEY_SIMPLEX,0.82,(255,0,0),2)
                    
                
                 cv.imshow("image",frame)

                 if cv.waitKey(1) & 0xFF == ord("q"):
                         break
             else:
                 print('cannot capture')
                 break
                 
         self.video.release()
         cv.destroyAllWindows()
            
if __name__ == "__main__":
    fr = Facerecognition(0,"haarcascade_frontalface_default.xml")
    fr.video_show_face()        


                 


    