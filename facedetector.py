#Step 0 - Import OpenCV
import cv2

#Step 1 - Create Classified Object
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Step 2 - Read the Image
img = cv2.imread("photo.jpg")

#Step 3 - Convert Image to GrayScale
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Step 4 - Detect faces
faces = face_cascade.detectMultiScale(grayimg, 
scaleFactor=1.05,
minNeighbors=5)

#Step 5 - Create Rectangle on all the faces
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

#Step 6 - Resize the Image
resizedimg = cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)))

#Step 7 - Show Resized & Rectangle Marked Image
cv2.imshow("Defected Face", resizedimg)

#Step 8 - Wait & Destory CV2
cv2.waitKey(0)
cv2.destroyAllWindows()
