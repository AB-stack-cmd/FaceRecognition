import mediapipe as mp
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv


video = cv.VideoCapture(0)
model_path = "hand_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


while True:
  ret,frame = video.read()
  if ret:
    base_options = BaseOptions(model_asset_path=model_path)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    cv.imshow('video',mp_image)
  if cv.waitKey(1) == ord('q'):
    break

video.release()
cv.destroyAllWindows()