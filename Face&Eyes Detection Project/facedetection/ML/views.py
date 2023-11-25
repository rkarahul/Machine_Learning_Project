from django.shortcuts import render
import cv2

# Create your views here.
def home(request):
    return render(request,"index.html")


def face(request):

    vdo=cv2.VideoCapture(0)
    model_face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    model_eye=cv2.CascadeClassifier("haarcascade_eye.xml")
    while True:
        isImg,img=vdo.read()
        if isImg==False:
            break
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=model_face.detectMultiScale(gray)
        eyes=model_eye.detectMultiScale(gray)
        for x,y,w,h in eyes: 
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,1),1)
        
        for x,y,w,h in faces: 
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

        cv2.imshow("img",img)
        key=cv2.waitKey(50)
        if key==ord('c'):
            break
    cv2.destroyAllWindows()
    vdo.release()
    return render(request,"face.html")