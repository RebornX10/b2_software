import cv2 as cv


# Setting up the video feed and the Window dimensions
capture = cv.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)
capture.set(10, 70)

# Rock/Paper/Scissors name attrinbution
classNames = []
classFile = 'RPS_names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Model Path
configPath = 'train.pbtxt'
modelPath = 'saved_model.pb'

# Detection of the model thru the video feed
net = cv.dnn_DetectionModel(modelPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = capture.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv.rectangle(img,box,color=(0,255,0),thickness=2)
            cv.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        c
                        0v.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv.imshow("RPS", img)
    cv.waitKey(1)
