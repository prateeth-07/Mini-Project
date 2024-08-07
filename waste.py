from cvzone.ClassificationModule import Classifier
import cv2

# url=""
cap = cv2.VideoCapture(0)
classifier = Classifier('D:\Puppy\bin-main\bin-main\keras_model (1).h5',"D:\Puppy\bin-main\bin-main\labels (1).txt")
while True:
    _, img = cap.read()
    pred =classifier.getPrediction(img)
    print(pred)
    cv2.imshow("image",img)
    if cv2.waitKey(2) & 0xff == ord('a'):
        break

    