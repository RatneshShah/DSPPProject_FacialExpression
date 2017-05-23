import cv2, glob, random, math, numpy as np, dlib, itertools
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from confusion_mat import plot_confusion_matrix
import matplotlib.pyplot as plt
from cv2 import *
import cPickle as pickle
import warnings

warnings.filterwarnings("ignore")

#emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"] #Emotion list
emotions = ["anger", "disgust", "happy", "neutral", "surprise"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
kernel = 'linear'

def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist) #Get the mean of both axes to determine centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist] #get distance between each point and the central point in both axes
        ycentral = [(y-ymean) for y in ylist]

        if xlist[26] == xlist[29]: #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
            anglenose = 0
        else:
            anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)

        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
            landmarks_vectorised.append(dist)
            #landmarks_vectorised.append(anglerelative)
    if len(detections) < 1:
        landmarks_vectorised = "error"
    return landmarks_vectorised
def detect_faces(gray):
    global faceDet,faceDet2,faceDet3,faceDet4
    #Detect face using 4 different classifiers
    face =  faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

    #Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face2) == 1:
        facefeatures = face2
    elif len(face3) == 1:
        facefeatures = face3
    elif len(face4) == 1:
        facefeatures = face4
    else:
        facefeatures = ""

    #Cut and save face
    for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
        print "face found:"
        gray = gray[y:y+h, x:x+w] #Cut the frame to size
        return gray
        try:
            out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
            return out
        except Exception as e:
            pass
    return gray
"""
Expression Detection on VIDEO
"""
kernel = 'linear'
# loading model
with open('svm_'+kernel+'.pkl', 'rb') as fid:
    clf = pickle.load(fid)

# Count variable to count the number of undetected faces
count = 0
file_type = 'JPG'
output = []
no_of_images = 4
for i in range(0, no_of_images):
    image = cv2.imread('test/test'+str(i)+'.'+file_type)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_image = clahe.apply(gray)
    landmarks_vectorised = get_landmarks(clahe_image)
    landmarks_vectorised = np.array(landmarks_vectorised)
    if landmarks_vectorised == "error":
        print "error"
    else:
        width = 0.35
        N = len(emotions)
        index = np.arange(N)

        pred = clf.predict(landmarks_vectorised)
        pred_prob = clf.predict_proba(landmarks_vectorised)
        print pred_prob
        fig, ax = plt.subplots()

        emotions = ("anger", "disgust", "happy", "neutral", "surprise")
        colors = ['#624ea7', 'g', 'yellow', 'k', 'maroon']
        plt.bar(index, pred_prob[0], width, color=colors)
        plt.title("Emotion Probabilty Histogram")
        plt.xlabel("Emotion Index")#+ "".join(emotions))
        plt.ylabel("Probabilty")

        plt.xticks(index,emotions)
        plt.tight_layout()
        plt.savefig('test/'+str(i)+'.png')
        #plt.show()
        output.append(emotions[pred[0]])
print output
