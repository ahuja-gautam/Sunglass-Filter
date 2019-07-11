
# coding: utf-8

# In[ ]:



import cv2
import numpy as np
import face_recognition



# In[ ]:

'''
def aspect_ratio(left, right):
    
    left=[np.array(x) for x in left]
    right=[np.array(x) for x in right]
    
    A1=np.linalg.norm(left[1]-left[5])
    B1=np.linalg.norm(left[2]-left[4])
    C1=np.linalg.norm(left[0]-left[3])
    
    ar1=100*(A1+B1)/C1
    
    A2=np.linalg.norm(right[1]-right[5])
    B2=np.linalg.norm(right[2]-right[4])
    C2=np.linalg.norm(right[0]-right[3])
    
    ar2=100*(A2+B2)/C2
    
    return round(ar1,3), round(ar2, 3)
'''

# In[ ]:



glasspts1=np.float32([[0,180], [600,180], [300, 300]])
glasspts2=np.float32([[30, 200], [600, 200], [320, 300]])
glasspts3=np.float32([[90, 210], [540, 210], [320, 320]])

glasses=[]

for i in range(1,4):
    glasses.append(cv2.resize((cv2.imread('glass%d.png'%i, cv2.IMREAD_UNCHANGED)), (640,480)))


glasspts=[glasspts1, glasspts2, glasspts3]


def transform(pts, idx):
    M=cv2.getAffineTransform(glasspts[idx], pts)
    dst=cv2.warpAffine(glasses[idx], M, (640, 480))
    ##dst=cv2.flip(dst,1)
    return dst.astype(float)


# In[ ]:


glass_idx=int(input("Enter the glass type: 0:Thug Life, 1:Shades1, 2:Shades2"))

print("Opening webcam, Press 'q' to exit")


# In[ ]:


##aspect_ratios=[]
##xvec=[]
## k=0
cap= cv2.VideoCapture(0)
while (True):
    ret , frame=cap.read()
    frame=cv2.flip(frame,1)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    frame=frame.astype(float)
    ##face_locations=face_recognition.face_locations(gray)
    face_landmarks_list = face_recognition.face_landmarks(gray)
    framecrop=np.ones((20,20))
    locations=[]
    pts=[]
    
    for landmark in face_landmarks_list:
        '''
        To get bounding boxes of the face
        top, right, bottom, left = loc
        
        locations.append(top)
        locations.append(right)
        locations.append(bottom)
        locations.append(left)
        '''
        lefteye=landmark['left_eye']
        righteye=landmark['right_eye']
        
        chin=landmark['chin']
        nose=landmark['nose_bridge']
        ##nose=nose[0]
    
        pts.append(chin[0])
        pts.append(chin[16])
        pts.append(nose[2])
        pts=np.float32(pts)
        ## nosepts=(np.array(nosepts).reshape((-1,1,2)))

        ##for cnt in nosepts:
          ##  cv2.draw(frame, [cnt], -1, (0,0,255), -1)
        
        glasstemp=transform(pts, glass_idx)
        alpha=glasstemp[:,:,3]/255.0
        glasstemp=glasstemp[:,:,0:3]
        for c in range(3):
            frame[:,:,c]=cv2.multiply(1-alpha,frame[:,:,c])
        for c in range(3):
            glasstemp[:,:,c]=cv2.multiply(alpha, glasstemp[:,:,c])
        frame=cv2.add(frame, glasstemp)
        
    
        '''
        ## for drawing outlines of the eyes
        eye1=cv2.convexHull(np.array(lefteye))
        eye2=cv2.convexHull(np.array(righteye))
        cv2.drawContours(frame, [eye1], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [eye2], -1, (0, 255, 0), 1)
        
        '''
        
    
        
        '''
        ##This gets aspect ratio of the eyes which indicates how open they are
        ##Useful for drowsiness detection
        ar1, ar2=aspect_ratio(lefteye, righteye)
        aspect_ratios.append((ar1+ar2)/2.0)
        xvec.append(k)
        k+=1
        '''


        '''
        for (x, y) in lefteye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in righteye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1) 
        for (x,y) in chin:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x,y) in nose:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1) 
        
        '''
            
        ## cv2.rectangle(frame, (left,top), (right, bottom), (0,255,0),1) 
        ## cv2.putText(frame, "ar1={} ar2={}".format(*[ar1,ar2]), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    

    cv2.imshow('frame',frame/255)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        
cap.release()
cv2.destroyAllWindows()
    


# In[ ]:


## plt.plot(xvec, aspect_ratios, 'r')

