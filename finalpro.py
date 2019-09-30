#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as clstr
import collections
from time import time
from collections import defaultdict
from scipy.ndimage import imread
from functools import partial
from sklearn.utils import shuffle
import os, glob,  shutil
import cv2 as cv2
from skimage.transform import resize
from scipy.stats import wasserstein_distance
from skimage.measure import compare_ssim
import imutils
from skimage.transform import resize
from scipy.stats import wasserstein_distance
import warnings
import re

if not os.path.exists('game'):
        os.makedirs('game')
def FrameCapture(path): 
    vidObj = cv2.VideoCapture(path) 
    count = 0
    success = 1
    while success: 

        success, image = vidObj.read() 

        cv2.imwrite("game/frame%d.jpg" % count, image) 

        count += 1
  
# Driver Code 
if __name__ == '__main__': 
        FrameCapture("game.mp4") 


# In[2]:




SQUARE_SIDE_LENGTH = 227
categories = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'empty', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']


# In[3]:


def cropb(img1):
    l2=[]     
    img=cv2.imread(img1)
    gray=cv2.imread(img1,0)
    ret,thresh = cv2.threshold(gray,127,255,1)     
    _,contours,_ = cv2.findContours(thresh,1,2)
    for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            if len(approx)==4:
                (x, y, w, h) = cv2.boundingRect(cnt)
                x1=w
                x2=h
                if((w*h)>500 and (h>20) and (w>20)):
                    l2.extend((x,y))


    x=int(l2[0])
    y=int(l2[1])
    w1=457
    w2=457
    crop_img = img[y:y+w1, x:x+w2]
    width = 457
    height = 457
    dim = (width, height)
    imageA = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
    cv2.rectangle(imageA,(2,2),(455,455),(0,0,0),2)
    return(imageA)



# In[4]:


dirListing = os.listdir('game')
dirFiles = []
if not os.path.exists('crop'):
        os.makedirs('crop')
for item in dirListing:
    if ".jpg" in item:
        dirFiles.append(item)
dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
cou=0
dk=[]
finl=[]
for i in dirFiles:
        
        j=str('game/'+str(i))
        statinfo=os.stat(j)
        if(statinfo.st_size>0):    

            im=cropb(j)
            cv2.imwrite("crop/crop%d.jpg" %cou,im )

            dk.append("crop/crop%d.jpg"%cou)
            finl.append("crop/crop%d.jpg"%cou)
            cou=cou+1


# In[5]:


from skimage.measure import compare_ssim

li=[]
k=0
k2=0
c2=[]
k1=0
for i in range(len(finl)-1):
        s1=cv2.imread(finl[i])
        s2=cv2.imread(finl[i+1])
        grayA = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        if(score>0.9999):
            k2=k2+1
        if(score<0.9999):
            k2=0
        if(k2==2):
            c2.append(finl[i+1])


# In[6]:


def centroids(imageA):
    
    gray = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
    mask = np.zeros((gray.shape),np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
    div = np.float32(gray)/(close)
    res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
    res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)
    thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)
    
    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,8))
    dx = cv2.Sobel(gray,cv2.CV_16S,1,0)
    dx = cv2.convertScaleAbs(dx)
    cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
    ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close,cv2.MORPH_ERODE,kernelx,iterations = 1)

    _,contour,_= cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if h/w > 15:
            cv2.drawContours(close,[cnt],0,255,-1)
        else:
            cv2.drawContours(close,[cnt],0,0,-1)
    close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
    kernel = np.ones((50,1), np.uint8)  
    d_im = cv2.dilate(close, kernel, iterations=20)
    e_im = cv2.erode(d_im, kernel, iterations=4) 
    closey = e_im.copy()
    
    
    
    
    kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(8,2))
    dy = cv2.Sobel(gray,cv2.CV_16S,0,1)
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
    ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close,cv2.MORPH_ERODE,kernely,iterations = 1)
    _,contour,_= cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if w/h > 25:
            cv2.drawContours(close,[cnt],0,255,-1)
        else:
            cv2.drawContours(close,[cnt],0,0,-1)
    close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
    kernel = np.ones((1,20), np.uint8)  # note this is a horizontal kernel
    d_im = cv2.dilate(close, kernel, iterations=7)
    e_im = cv2.erode(d_im, kernel, iterations=5) 
    closey1 = e_im.copy()    
    
    
    res1 = cv2.bitwise_and(closey,closey1)
    kernel = np.ones((3,3),np.uint8)
    sure_bg = cv2.dilate(res1,kernel,iterations=3)

    dist_transform = cv2.distanceTransform(res1,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)


    _,contour,_= cv2.findContours(unknown,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contour:
        mom = cv2.moments(cnt)
        if(mom['m00']!=0):
            (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
            centroids.append((x,y))


    centroids = np.array(centroids,dtype = np.float32)
    c = centroids.reshape((81,2))
    c2 = c[np.argsort(c[:,1])]

    b = np.vstack([c2[i*9:(i+1)*9][np.argsort(c2[i*9:(i+1)*9,0])] for i in range(9)])
    bm = b.reshape((9,9,2))
        
    cc=0
    
    
    '''uncomment to get cropped images of all 64 squares
    for i in range(8):
        for j in range(8):
            #k1=cv2.rectangle(imageA,(bm[i][j][k],bm[i][j][k+1]),(bm[i+1][j+1][k],bm[i+1][j+1][k+1]),(0,255,0),3)
             x=int(bm[i][j][0])
             y=int(bm[i][j][1])
             w1=54
             w2=54
             crop_img = imageA[y:y+w1, x:x+w2]
             cv2.imwrite("temp1/rect%d.jpg" %cc,crop_img)
             cc=cc+1
    '''
    return(bm)
    


# In[7]:


def labelsquare(bm):
    li1=['a','b','c','d','e','f','g','h']
    li2=['8','7','6','5','4','3','2','1']

    l=[]
    k=[]
    n=0
    for i in range(8):
        for j in range(8):
             p=li1[i]+li2[j]
             kl=bm[j][i]
             k.append(p)
             l.append(kl)
    dict = {k: v for k, v in zip(k,l)}    
    bm=dict
    i=0
    j=0 
    k=0
    
    return(dict)


# In[8]:


warnings.filterwarnings('ignore')
height = 2**10
width = 2**10



def get_img(path, norm_size=True, norm_exposure=False):
  img = imread(path, flatten=True).astype(int)
  if norm_size:
    img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)
  if norm_exposure:
    img = normalize_exposure(img)
  return img
def get_histogram(img):
 
  h, w = img.shape
  hist = [0.0] * 256
  for i in range(h):
    for j in range(w):
      hist[img[i, j]] += 1
  return np.array(hist) / (h * w) 
def normalize_exposure(img):
  img = img.astype(int)
  hist = get_histogram(img)
  cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
  sk = np.uint8(255 * cdf)
  height, width = img.shape
  normalized = np.zeros_like(img)
  for i in range(0, height):
    for j in range(0, width):
      normalized[i, j] = sk[img[i, j]]
  return normalized.astype(int)
def earth_movers_distance(path_a, path_b):
  img_a = get_img(path_a, norm_exposure=True)
  img_b = get_img(path_b, norm_exposure=True)
  hist_a = get_histogram(img_a)
  hist_b = get_histogram(img_b)
  return wasserstein_distance(hist_a, hist_b)

def check(img_a):
      li=['temp1/blank1.jpg','temp1/wpawn.jpg','temp1/bpawn.jpg','temp1/wrook.jpg','temp1/brook.jpg','temp1/wknight.jpg','temp1/bknight.jpg','temp1/wbishop.jpg','temp1/bbishop.jpg','temp1/wqueen.jpg','temp1/bqueen.jpg','temp1/wking.jpg','temp1/bking.jpg']
      low2=999
      c=0  
      for j in li:
          emd=0.0
          img_b=str(j)
          emd = earth_movers_distance(img_a,  img_b)
          if(emd<low2):
                low2=emd
                k=j
      
      return(k) 


# In[10]:


count1=1
count2=1
co1=''
co2=''
s=''
co=0
lo=[]
li=[]
for i in range(len(c2)-1):
    bn=''
    su=0
    image1=cv2.imread(c2[i])
    image2=cv2.imread(c2[i+1])

    bm1=centroids(image1)
    bm2=centroids(image2)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    if(score<0.999):
       
        lo.append(c2[i+1])
        
        thresh = cv2.threshold(diff, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        l2=[]
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if((w*h)>800 and (h>30) and (w>30)):
                su=su+1
                l2.extend((x,y))
        if(su>1 and su<4):
            min1=999
            min2=999
            p1=l2[0]
            p2=l2[1]
            p3=l2[2]
            p4=l2[3]
            li1=[]
            li2=[]
            for i in range(8):
                for j in range(8):
                    z=0
                    q=0
                    z1=abs(bm1[i][j][0]-p1)
                    z2=abs(bm1[i][j][1]-p2)
                    z3=abs(bm2[i][j][0]-p3)
                    z4=abs(bm2[i][j][1]-p4)
                    q=z3+z4
                    z=z1+z2
                    if(z<min1):
                        min1=z
                        m1=bm1[i][j][0]
                        m2=bm1[i][j][1]
                    if(q<min2):
                        min2=q
                        m3=bm2[i][j][0]
                        m4=bm2[i][j][1]
            li1.extend((m3,m4))
            li2.extend((m1,m2))
            dict1=labelsquare(bm1)
            dict2=labelsquare(bm2)
            lok1 = [key  for (key, value) in dict1.items() if  list(li1)==list(value)]
            lok2 = [key  for (key, value) in dict2.items() if  list(li2)==list(value)]
            if(count1%2==0):
                a=dict1.get(str(lok1[0])).tolist()
                b=dict2.get(str(lok2[0])).tolist()
                xn=str(lok2[0])
                xm=str(lok1[0])
                print(lok1,lok2)
                

            else:
                a=dict1.get(str(lok2[0])).tolist()
                b=dict2.get(str(lok1[0])).tolist()
                xn=str(lok1[0])
                xm=str(lok2[0])
                print(lok2,lok1)

            x1=int(a[0])
            y1=int(a[1])
            x2=int(b[0])
            y2=int(b[1])
            w=54
            h=54
            if not os.path.exists('game1'):
                os.makedirs('game1')
            crop_img1 = image1[y1:y1+h, x1:x1+w]
            crop_img2 = image1[y2:y2+h, x2:x2+w]
            cv2.imwrite("game1/dd%d.jpg" %count1,crop_img1)
            cv2.imwrite("game1/de%d.jpg" %count2,crop_img2)

            co1=str("game1/dd"+str(count1)+".jpg")
            co2=str("game1/de"+str(count2)+".jpg")
            count1=count1+1
            count2=count2+1
           
            x1=check(co1)
            x2=check(co2)
            if(x1=='temp1/wpawn.jpg' or x1 =='temp1/bpawn.jpg'):
                bn=''
            if(x1=='temp1/wrook.jpg' or x1 == 'temp1/brook.jpg'):
                bn='R'
            if(x1=='temp1/wbishop.jpg' or x1 == 'temp1/bbishop.jpg'):
                bn='B'
            if(x1== 'temp1/wknight.jpg' or x1 == 'temp1/bknight.jpg'):
                bn='N'
            if(x1== 'temp1/wqueen.jpg' or x1== 'temp1/bqueen.jpg'):
                bn='Q'
            if(x1== 'temp1/wking.jpg' or x1== 'temp1/bking.jpg'):
                bn='K'  
            if(su==2 or su==3):
                if(x2=='temp1/blank1.jpg'):
                    s=bn+xn
                    print(bn+xn)
                elif((x1=='temp1/bpawn.jpg'or x1=='temp1/wpawn.jpg') and x2!='temp1/blank1.jpg'):
                    print(xm[0]+"x"+xn)
                    s=xm[0]+"x"+xn
                elif(x2!='temp1/blank1.jpg'and (x1!='temp1/wpawn.jpg'or x1!='temp1/bpawn.jpg')):
                    s=bn+"x"+xn
                    print(bn+"x"+xn)
                li.append(s)
        if(su==4):
            count1=count1+1
            print("0-0")
            li.append("0-0")
        
        if(su>4):
            count1=count1+1
            print("0-0-0")
            li.append("0-0-0")
         
    


# In[ ]:





# In[11]:


k=0
for i in range(len(li)):
    if(i%2==0):
        k=k+1
        k1=str(k)+")"
        with open("Output.txt", "a") as text_file:
            text_file.write(k1+' '+li[i]+' ')
    else:
        with open("Output.txt", "a") as text_file:
            text_file.write(li[i]+' '+'\n'+'\n')


# In[12]:


lo1=[]
numbers=0
for i in range (len(lo)-1):
    num=re.findall('\d+',lo[i])
    num = map(int,num)
    num1=re.findall('\d+',lo[i+1])
    num1=map(int,num1)
    nu=max(num)
    nu1=max(num1)
    if((nu1-nu)<7):
        continue
    elif(i==0):
        continue
    else:
        lo1.append(lo[i])
        
lo1.append(lo[-1:])


# In[13]:


li1=[]
for j in lo1:
    j1=str(j)
    j1=j1.split('/')
    j1=j1[1]
    j1=j1.replace('crop','frame')
    li1.append(j1)


# In[ ]:





# In[14]:



from os.path import isfile, join
p=[]
k=' '
pathIn= 'game'
img_array = []
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]#for sorting the file names properly
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
count=0
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.75
thickness = 1
color = (255,255,255)
x=500
y=100
c=0
for filename in files:
    st=str('game/'+filename)
    statinfo=os.stat(st)
    if(statinfo.st_size>0):
        img = cv2.imread('game/'+filename)
        cv2.rectangle(img,(500,10),(1200,700),(0,255,0),3)

        height, width = img.shape[:2]
        for j in li1:
            if(filename==j):
                x=500
                y=100
                k=li[c]
                c=c+1
                p.append(k)
                for j2 in p:
                    cv2.putText(img,j2,(x,y),font,font_scale, color, thickness)
                    x=x+80
                    if(x>1200):
                        y=y+40
                        x=500
            else:
                x=500
                y=100
                for j2 in p:
                    cv2.putText(img,j2,(x,y),font,font_scale, color, thickness)
                    x=x+80 
                    if(x>1200):
                        y=y+40
                        x=500

        img_array.append(img)
out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'DIVX'),15, (width,height))
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


# In[15]:


def check1(img):
    bn=0
    img_rgb6 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb6,cv2.COLOR_BGR2GRAY)
    king_white_template = cv2.imread("temp1/wking.jpg",0)    
    w_king_white, h_king_white = king_white_template.shape[::-1]
    res_king_white = cv2.matchTemplate(img_gray,king_white_template,cv2.TM_CCOEFF_NORMED)
    threshhold = 0.6
    loc = np.where(res_king_white >= threshhold)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    color = (0, 0, 169)
    ll=[]
    k=[]
    for pt in zip(*loc[::-1]):
        ll.append(pt)

    x1=0
    x2=0
    for i in ll :
        x=0
        j=list(i)
        ll1=tuple(j)
        k1=abs(j[0]-x1)
        k2=abs(j[1]-x2)

        k=k1+k2
        if(k>4):
            p1=ll1[0]
            p2=ll1[1]
        x1=j[0]
        x2=j[1]
    lk1=[]
    min1=10
    for i in range(8):
                for j in range(8):
                        z=0
                        bm1=centroids(img)
                        z1=abs(bm1[i][j][0]-p1)
                        z2=abs(bm1[i][j][1]-p2)
                        z=z1+z2
                        if(z<min1):
                            min1=z
                            m1=bm1[i][j][0]
                            m2=bm1[i][j][1]

    lk1.extend((m1,m2))
    dict1=labelsquare(bm1)
    lok1 = [key  for (key, value) in dict1.items() if  list(lk1)==list(value)]
    s=str(lok1[0])
    k=int(s[1])-1
    cou=0
    for i in range(k,0,-1):
        if(i<k):
            x=str(s[0]+str(i))
            h=54
            w=54
            x1=dict1.get(x).tolist()
            z1=int(x1[0])
            z2=int(x1[1])
            crop_img1 = img[z2:z2+h, z1:z1+w]
            cv2.imwrite("ga/dd%d.jpg"%cou ,crop_img1)
            sk="ga/dd%d.jpg"%cou
            xk=check(sk)
            if(xk!='temp1/blank1.jpg'):
                if(xk=='temp1/wrook.jpg' or xk=='temp1/wqueen.jpg'):
                    bn=1
                    break
            cou=cou+1
    for j in range(k,9):
        if(i>k):
            x=str(s[0]+str(i))
            h=54
            w=54
            x1=dict1.get(x).tolist()
            z1=int(x1[0])
            z2=int(x1[1])
            crop_img1 = img[z2:z2+h, z1:z1+w]
            cv2.imwrite("ga/dd%d.jpg"%cou ,crop_img1)
            sk="ga/dd%d.jpg"%cou
            xk=check(sk)
            if(xk!='temp1/blank1.jpg'):
                if(xk=='temp1/wrook.jpg' or xk=='temp1/wqueen.jpg'):
                    bn=1
                    break
            cou=cou+1
    
    return(bn)  
                                           
    
                                     





# In[21]:


import os
import sys
import shutil
try:
    shutil.rmtree("game")
    shutil.rmtree("game1")
    shutil.rmtree("crop")

except OSError as e:
    print ("Error: %s - %s." % (e.filename, e.strerror))


# In[ ]:





# In[ ]:




