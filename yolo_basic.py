import cv2
import numpy as np

classes_File='coco name'
classnames=[]
with open(classes_File,'rt') as f:
    classnames=f.read().rstrip('\n').split('\n')
print(classnames)
dim=120
conf_threshold=0.5
nms_threshold=0.3
modelConfig='yolov3-tiny.cfg'
modelweights='yolov3-tinyh'

net=cv2.dnn.readNetFromDarknet(modelConfig, modelweights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

cap=cv2.VideoCapture(0)

def findObjects(outputs,img):
    height,width,channels=img.shape
    bounding_box_list=[]
    classId_list=[]
    confidence_list=[]
    for output in outputs:
        for detection in output:
            scores=detection[5:]
            classId=np.argmax(scores)
            confidence=scores[classId]
            if confidence>0.4:
                w,h=int(detection[2]*width),int(detection[3]*height)
                x,y=int((detection[0]*width)-w/2),int((detection[1]*height)-h/2)
                bounding_box_list.append([x,y,w,h])
                classId_list.append(classId)
                confidence_list.append(float(confidence))
    print(len(bounding_box_list))
    indices=cv2.dnn.NMSBoxes(bounding_box_list,confidence_list,0.4,0.1)
    for i in indices:
        i=i[0]
        box=bounding_box_list[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classnames[classId_list[i]]} {int(confidence_list[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)

while True:
    success,img=cap.read()

    blob_img=cv2.dnn.blobFromImage(img,1/255,(dim,dim),[0,0,0],crop=False)
    net.setInput(blob_img)

    layer_names=net.getLayerNames()
    output_names=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs=net.forward(output_names)
    findObjects(outputs,img)



    cv2.imshow("Image",img)
    cv2.waitKey(1)
