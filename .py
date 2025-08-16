import cv2 as cv

class Facerecognition:
    'came for camera used 0 for pc cam and 1 for external cam'
    'cascade for recognition of face or eye or object or more'

    def __init__(self, came, cascade_path):
        self.video = cv.VideoCapture(came)
        self.cascade_path = cv.CascadeClassifier(cv.data.haarcascades + cascade_path)

    def video_capture_gray(self, file_name, show=False):
        while True:
            ret, frame = self.video.read()
            if ret:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                cv.imwrite(file_name, gray)
                print("Captured gray image")

                if show:
                    cv.imshow("Gray Image", gray)
                    if cv.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    break
        self.video.release()
        cv.destroyAllWindows()

    def video_show_face(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                print("Failed to grab frame")
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = self.cascade_path.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))

            for (x, y, w, h) in faces:
                # Rectangle around face
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Label above rectangle
                cv.putText(frame, "Face Detected", (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.82, (255, 0, 0), 2, cv.LINE_AA)

            cv.imshow("Face Detection", frame)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        # Release after loop
        self.video.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    fr = Facerecognition(0, "haarcascade_frontalface_default.xml")
    fr.video_show_face()
