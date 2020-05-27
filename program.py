import numpy as np
import cv2

"""
Részek
1,Létrehozom a tömböt amiben tárolom és feltöltöm színnel és meghatározom egy betűvel
2,Fekete részét a kockának kiemelemm, utána a kockát felismerem és eltárolom a 4 sarkát
3,Kiszámolom a 4 sarok alapján a 9 kockának milyen színe van majd feltöltöm a Cube végeredmény tömböt
"""

#1. rész
#ebben tárolja majd a színeket, ez lesz a végeredmény tömb
Cube = np.zeros((6, 3, 3),dtype='U')
#itt minden 3*3-as mátrixnak megadjuk melyik színű oldalt képviseli,ez alapján azonosítja majd a program
for i in range(6):
    if i==0:
        Betu="R"
    if i==1:
        Betu="W"
    if i==2:
        Betu="O"
    if i==3:
        Betu="Y"
    if i==4:
        Betu="G"
    if i==5:
        Betu="B"
    Cube[i][1][1]=Betu


"""
2.rész
itt fog majd a képek száma alapján beolvasni
fontos hogy 0 tól legyenek számozva a képek és jpg formátúmbaman
képek darabszámát kell beírni a következő sorban a kettes helyére
amennyiben más a képformátúm a .jpg-t át kell írni
"""
for M in range(2):
    Pic=str(M)+str(".jpg")
    #Kiemelem a fekete színt
    img2 = cv2.imread(Pic)
    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)


    #Ezzel az intervalummal találta meg  a feketét
    lower_range = np.array([0,0,0])
    upper_range = np.array([244,104,150])
     
    mask = cv2.inRange(hsv, lower_range, upper_range)

    cv2.imshow('mask', mask)
    cv2.imwrite("F1mask1.png", mask)


    #Most bejelölöm a négyzeteket 
    w2=0
    h2=0
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
            if w > w2 and h>h2 and aspectRatio >= 0.50 and aspectRatio <= 1.5:
                w2=w
                h2=h
                t=approx

    #3. rész ,számolás
                
    CN=[]
    sz=[1,2,3,4,5,6] 
    for i in range(3):

    #itt kiszámolom a két oldal közepét és eltárolom
        if i==0:
            c=2
            # A a kocka közepét adja ki 
        if i==1:
            c=8
            #A kocka felső részét adja ki
        if i==2:
            c=8/7
            #A kocka alsó részét adja ki
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
                x2=x2+1
                
            #a szín lekérdezése /azonosítása/tárolása CN-ben
            #más színek esetén át kell írni az RGB értékeket az ifekben
            color=img2[sz[5]][sz[4]]
            interv=10
            while interv<100:
                interv=interv+5
                #piros
                if color[2]>=125-interv and color[2]<=125+interv and color[1]>=35-interv and color[1]<=35+interv and color[0]>=35-interv and color[0]<=35+interv  :
                    
                    CN.append("R")
                    break
                #sarga
                if color[2]>=170-interv and color[2]<=170+interv and color[1]>=180-interv and color[1]<=180+interv and color[0]>=60-interv and color[0]<=60+interv  :

                    CN.append("Y")
                    break
                #narancs
                if color[2]>=200-interv and color[2]<=200+interv and color[1]>=80-interv and color[1]<=80+interv and color[0]>=70-interv and color[0]<=70+interv  :
                
                    CN.append("O")
                    break
                #kék
                if color[2]>=20-interv and color[2]<=20+interv and color[1]>=55-interv and color[1]<=55+interv and color[0]>=100-interv and color[0]<=100+interv  :
                
                    CN.append("B")
                    break
                #zöld
                if color[2]>=20-interv and color[2]<=20+interv and color[1]>=135-interv and color[1]<=135+interv and color[0]>=40-interv and color[0]<=40+interv  :
                
                    print("zöld")
                    CN.append("G")
                    break           
                #fehér
                if color[2]>=160-interv and color[2]<=160+interv and color[1]>=155-interv and color[1]<=155+interv and color[0]>=150-interv and color[0]<=150+interv  :
                
                    CN.append("W")
                    break

    # itt CN segítségével feltölti az oldalt
    for i in range(6) :
        if CN[0] == Cube[i][1][1]:
            
            Cube[i][1][1]=CN[0]
            Cube[i][1][0]=CN[1]
            Cube[i][1][2]=CN[2]
            
            Cube[i][0][1]=CN[3]
            Cube[i][0][0]=CN[4]
            Cube[i][0][2]=CN[5]
            
            Cube[i][2][1]=CN[6]
            Cube[i][2][0]=CN[7]
            Cube[i][2][2]=CN[8]
            
print(Cube);
cv2.waitKey(0)
cv2.destroyAllWindows()
