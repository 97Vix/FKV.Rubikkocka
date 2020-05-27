
import numpy as np
import cv2

#Kiemelem a fekete színt 

img2 = cv2.imread('F1.jpg')
hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)


#Ezzel az intervalummal találta meg  a feketét
lower_range = np.array([0,0,0])
upper_range = np.array([244,104,150])
 
mask = cv2.inRange(hsv, lower_range, upper_range)

cv2.imshow('mask', mask)
cv2.imwrite("F1mask1.png", mask)



#2-------------------------------------------------------------

#Most bejelölöm a négyzeteket 
img = cv2.imread('F1mask1.png')
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (169, 100, 100), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 4:
        x1 ,y1, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        if w >500 and h>500: #ez még át lesz alakítva
            t=approx
            print("approx: ", approx)

#számolás
sz=[1,2,3,4,5,6] 
print(sz)
for i in range(3):

#itt kiszámolom a két oldal közepét és eltárolom
    if i==0:
        c=2
        print(" A a kocka közepe")
    if i==1:
        c=8
        print("A kocka felső része")
    if i==2:
        c=8/7
        print("A kocka alsó része")
    for x in range(4):
        if x == 0:
            k1=0
            k2=1
            x2=0
        if x == 2:
            k1=k1+3
            k2=k2+1
        if x2 == 2:
            x2=0
            
        a=t[k1][0][x2]
        b=t[k2][0][x2]
        sz[x]=int(a+(b-a)/c)
        x2=x2+1
    #kocka közepe szamítas, a két oldal közepéből elsőnek ,utána majd a bal majd a jobb oldal kordinátáját
    for y in range(3):
        if y==0:
            c=2
        if y==1:
            c=8
        if y==2:
            c=8/7
        for x in range(2):
            if x == 0:
                k1=0
                k2=2
            if x == 1:
                k1=k1+1
                k2=k2+1
            if x2 == 2:
                x2=0
                
            a=sz[k1]
            b=sz[k2]
            x=x+4
            sz[x]=int(a+(b-a)/c)
            #print(sz)
            x2=x2+1
            
        #a szín lekérdezése /azonosítása/tárolása lesz majd
        print("az ott taláhtaó színkód:", img2[sz[5]][sz[4]])
            
        
#vege
cv2.waitKey(0)
cv2.destroyAllWindows()
