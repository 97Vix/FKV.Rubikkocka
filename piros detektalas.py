import numpy as np
import cv2

#Itt Detektálom a pirosat

img2 = cv2.imread('F1.jpg')
hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

#Ezzel az intervalummal találta meg a legjobban a pirosat 
lower_range = np.array([0,120,100])
upper_range = np.array([179,200,155])
 
mask = cv2.inRange(hsv, lower_range, upper_range)

cv2.imshow('mask', mask)
#csak lemetett képpel működött
cv2.imwrite("F1mask1.png", mask)



#Innetől kontúrt rakok a négyzetekre
img = cv2.imread('F1mask1.png')
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (169, 100, 100), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5



#Ez a rész a kontúrt kiemeli hogy csak a négyzet vonalai maradjanak meg
#Talán túl vastag a kontúr, ezért ismeri fel a külső négyzetett és a belsőt ezzel duplázva a kordinátákat

img2 = img
hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
 

 
lower_range = np.array([50,50,150])
upper_range = np.array([150,150,180])
 
mask = cv2.inRange(hsv, lower_range, upper_range)


cv2.imshow('mask', mask)
cv2.imwrite("F1mask2.png", mask)



#Itt már újrakontúrozom és elkérem a négyzet kordinátáit
#egyenlőre még probléma hogy random sarkok kordinátáit kapom 

img = cv2.imread('F1mask2.png')
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
        
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
          cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 2500))
          print(x, y)
        else:
          cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255))
          print(x, y)
#négyzet és téglalap detektálás is kell ,mert mindkettőnek láthatja



cv2.imshow("shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



