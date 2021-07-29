# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 10:21:34 2021

@author: Nilesh
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Below are 4 sets of images defined.
Uncomment any one of the lines to generate mosaic of that set

"""
#images = ['Seta//image01.jpg','Seta//image02.jpg','Seta//image03.jpg','Seta//image04.jpg','Seta//image05.jpg']
#images = ['Seta//image03.jpg','Seta//image04.jpg']
images = ['Setd//1.jpeg','Setd//2.jpeg','Setd//3.jpeg','Setd//4.jpeg','Setd//5.jpeg']
#images = ['Setb//1.jpeg','Setb//2.jpeg','Setb//3.jpeg','Setb//4.jpeg','Setb//5.jpeg']



color_imgs = []
imgs = []
keypoints = []
descriptors = []

sift = cv2.xfeatures2d.SIFT_create(nfeatures = 5000)

counter = 0
gscaleX = 3024
gscaleY = 4024
gscaleX = 774
gscaleY = 1032

def interpolate(p2,img2):
    x1,y1 = np.floor(p2)
    return img2[int(y1),int(x1),:]

def transformH(p1,H,scaleX,scaleY,Xoffset = 0,Yoffset = 0):
    x1,y1 = p1[0],p1[1]
    p1 = np.array([x1,y1,1]).T
    p2_ = np.dot(H,p1)
    x1,y1 = scaleX*p2_[0]/p2_[2],scaleY*p2_[1]/p2_[2]
    return(x1,y1)

def scaled(rows1,cols1,rows2,cols2,cords1,cords2):
    c1,c2 = [],[]
    rows1 = gscaleY
    rows2 = gscaleY
    cols1 = gscaleX
    cols2= gscaleX
    for i in range(0,len(cords1)):
        x,y = cords1[i]
        x = x/cols1
        y = y/rows1
        c1.append([x,y])
        
    for i in range(0,len(cords2)):
        x,y = cords2[i]
        x = x/cols2
        y = y/rows2
        c2.append([x,y])
        
    return c1,c2
    
def computeH(pl1,pl2):
    A = []
    for i in range(0,len(pl1)):
        x1,y1 = pl1[i]
        x2,y2 = pl2[i]
        a = [[x1,y1,1,0,0,0,-x1*x2,-y1*x2,-x2],[0,0,0,x1,y1,1,-x1*y2,-y1*y2,-y2]]
        A.append(a)
    A = np.array(A)
    A = np.reshape(A,[2*len(pl1),9])
    t = A.T @ A
    w,v = np.linalg.eig(t)
    minidx = np.argmin(w)
    H = v[:,minidx]
    return H

def computeCost(H,p1,p2,scaleX,scaleY):
    x1,y1 = p1[0],p1[1]
    p1 = np.array([x1,y1,1]).T
    p2_ = np.dot(H,p1)
    x1,y1 = p2_[0]/p2_[2],p2_[1]/p2_[2]
    x2,y2 = p2[0],p2[1]
    c = scaleX*abs(x2-x1) + scaleY*abs(y2 - y1)
    return(c)

def readImages():
    imgs = []
    keypoints = []
    descriptors = []
    counter = 0
    plt.figure(figsize = [20,20])
    for imgpath in images:
        i = cv2.imread(imgpath)
        color_imgs.append(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
        gray = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
        imgs.append(gray)
        k, d = sift.detectAndCompute(gray, None)
        keypoints.append(k) 
        descriptors.append(d)
        # draw the detected key points
        #sift_image = cv2.drawKeypoints(gray, k, i,[255,0,0])
        # show the image
        #cv2.namedWindow("image{}".format(counter), cv2.WINDOW_NORMAL) 
        #cv2.imshow('image{}'.format(counter), sift_image)
        # save the image
        #cv2.imwrite("table-sift{}.jpg".format(counter), sift_image)
        # if counter < 2: 
        #     plt.subplot(1,2,counter+1)
        #     plt.imshow(i)
        # counter += 1
    return imgs,keypoints,descriptors

def getMatches(img1, img2,matcher):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures = 5000)
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    
    if matcher == 'bf':
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(d1,d2)
    else:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
    
        flann = cv2.FlannBasedMatcher(index_params,search_params)
    
        matches = flann.knnMatch(d1,d2,k=2)
    
    
    
    return matches,img1,img2,rows1,cols1,rows2,cols2,k1,k2
    
#feature matching
def returnMatches(i,j):
    global rows1, rows2, cols1, cols2
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    descriptors_1 = descriptors[i]
    descriptors_2 = descriptors[j]
    keypoints_1 = keypoints[i]
    keypoints_2 = keypoints[j]
    img1 = imgs[i]
    img2 = imgs[j]
    
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    
    matches = bf.match(descriptors_1,descriptors_2)
    #matches = sorted(matches, key = lambda x:x.distance)
    return matches,img1,img2,rows1,cols1,rows2,cols2,keypoints_1,keypoints_2

def getFeature_cords(matches,keypoints_1,keypoints_2):
    img1cords = []
    img2cords = []
    
    for i in range(0,len(matches)):
        trainIndex = matches[i].trainIdx
        queryIndex = matches[i].queryIdx
        kpq = keypoints_1[queryIndex]
        kpt = keypoints_2[trainIndex]
        pt1 = [int(x) for x in list(np.round(kpq.pt))]
        pt2 = [int(x) for x in list(np.round(kpt.pt))]
        img1cords.append(pt1)
        img2cords.append(pt2)
    
    return img1cords,img2cords

def RANSAC(matches,scaled_img1cords,scaled_img2cords):
    optratio = 0
    dist_thresh = 10
    
    for j in range(0,1000):
        indxlist = np.random.choice(np.arange(0,len(matches)),size = 4,replace = False)
        s_img1 = np.array(scaled_img1cords)
        s_img2 = np.array(scaled_img2cords)
        H = computeH(s_img1[list(indxlist)], s_img2[list(indxlist)])
        H = np.reshape(H,[3,3])
        cost = 0
        
        inliers = 0
        outliers = 0
        for i in range(0,len(matches)):
            cost = computeCost(H, scaled_img1cords[i], scaled_img2cords[i],gscaleX,gscaleY)
            #print(cost)
            if cost <= dist_thresh:
                inliers += 1
            else:
                outliers += 1
        ratio = inliers/(inliers + outliers)
        if ratio > optratio:
            optH = H
            optratio = ratio
            print(ratio)
            
    return optH

def Homographytransform(optH,img2):
    img4 = np.zeros([rows2,cols2+cols1])
    for y in range(0,rows2):
        print(100*y/rows2)
        for x in range(0,2*cols2):
            x1,y1 = transformH([x/cols2,y/rows2], optH,cols2,rows2)
            if x1 < 0 or y1 < 0:
                val = 0
            elif x1>=cols2 or y1>=rows2:
                val = 0
            else:
                val = interpolate([x1,y1], img2)
            img4[y,x] = val
    return img4

def Homographytransform_c(canvas,xmin,ymin,w,h,optH,img2,c_img2):
    #rows2 = canvas.shape[0]
    #cols2 = canvas.shape[1]
    rows2 = img2.shape[0]
    cols2= img2.shape[1]
    lastPercent = 0
    for y in range(ymin,ymin+h):
        percent = int(100*(y-ymin)/h)
        if percent != lastPercent:
            print(percent)
            lastPercent = percent
        for x in range(xmin,xmin+w):
            x1,y1 = transformH([x/gscaleX,y/gscaleY], optH,gscaleX,gscaleY)
            if x1 < 0 or y1 < 0:
                val = np.zeros(3)
            elif x1>=cols2 or y1>=rows2:
                val = np.zeros(3)
            else:
                val = interpolate([x1,y1], c_img2)
            canvas[y-ymin,x-xmin,:] = val
    return canvas

def getCanvas(img2,H):
    rows,cols = img2.shape[0],img2.shape[1]
    Hinv = np.linalg.inv(H)
    top_left = [0,0]
    top_right = [cols/gscaleX,0]
    bottom_left = [0,rows/gscaleY]
    bottom_right = [cols/gscaleX,rows/gscaleY]
    tl_x,tl_y = transformH(top_left, Hinv, gscaleX,gscaleY)
    tr_x,tr_y = transformH(top_right, Hinv, gscaleX,gscaleY)
    br_x,br_y = transformH(bottom_right, Hinv, gscaleX,gscaleY)
    bl_x,bl_y = transformH(bottom_left, Hinv, gscaleX,gscaleY)
    xmin = int(min(tl_x,bl_x))
    xmax = int(np.ceil(max(tr_x,br_x)))
    ymin = int(min(tl_y,tr_y))
    ymax = int(np.ceil(max(bl_y,br_y)))
    w = xmax - xmin
    h = ymax - ymin
    canvas = np.zeros([int(h),int(w),3])
    return canvas,xmin,ymin,w,h

        
def getHs():
    Hs = []
    for i in range(0,2):
        matches,img1,img2,rows1,cols1,rows2,cols2,keypoints_1,keypoints_2 = returnMatches(i+1,i)
        img1cords,img2cords = getFeature_cords(matches,keypoints_1,keypoints_2)
        scaled_img1cords , scaled_img2cords = scaled(rows1, cols1, rows2, cols2, img1cords, img2cords)
        print("Performing RANSAC on: ",i+1,i)
        optH = RANSAC(matches,scaled_img1cords,scaled_img2cords)
        Hs.append(optH)
    
    for j in range(2,4):
        matches,img1,img2,rows1,cols1,rows2,cols2,keypoints_1,keypoints_2 = returnMatches(j,j+1)
        img1cords,img2cords = getFeature_cords(matches,keypoints_1,keypoints_2)
        scaled_img1cords , scaled_img2cords = scaled(rows1, cols1, rows2, cols2, img1cords, img2cords)
        print("Performing RANSAC on: ",j,j+1)
        optH = RANSAC(matches,scaled_img1cords,scaled_img2cords)
        Hs.append(optH)
    #Hs = [Hs[0] @ Hs[1], Hs[1], Hs[2], Hs[3] @ Hs[2]]
    
    return Hs

def get_specific_wrap(img1,img2,c_img1,c_img2):
    matches,img1,img2,rows1,cols1,rows2,cols2,keypoints_1,keypoints_2 = getMatches(img1, img2,'bf')
    img1cords,img2cords = getFeature_cords(matches,keypoints_1,keypoints_2)
    scaled_img1cords , scaled_img2cords = scaled(rows1, cols1, rows2, cols2, img1cords, img2cords)
    
    optH = RANSAC(matches,scaled_img1cords,scaled_img2cords)
    canvas,xmin,ymin,w,h = getCanvas(img2, optH)
    img4 = Homographytransform_c(canvas,xmin,ymin,w,h,optH,img2,c_img2)
    return [canvas,xmin,ymin,w,h,img4] ,optH

def get_all_warped(Hs):
    canvasDict = {}
    
    for i in range(0,2):
        img2 = imgs[i]
        optH = Hs[i]
        canvas,xmin,ymin,w,h = getCanvas(img2, optH)
        print("Calculating warped image ",i)
        img4 = Homographytransform_c(canvas,xmin,ymin,w,h,optH,img2)
        canvasDict[i] = [canvas,xmin,ymin,w,h,img4]
        
    for j in range(3,5):
        img2 = imgs[j]
        optH = Hs[j-1]
        canvas,xmin,ymin,w,h = getCanvas(img2, optH)
        print("Calculating warped image ",j-1)
        img4 = Homographytransform_c(canvas,xmin,ymin,w,h,optH,img2)
        canvasDict[j] = [canvas,xmin,ymin,w,h,img4]
    
    return canvasDict

def get_wrap(optH,img2):
    canvasDict = {}
    canvas,xmin,ymin,w,h = getCanvas(img2, optH)
    img4 = Homographytransform_c(canvas,xmin,ymin,w,h,optH,img2)
    canvasDict[0] = [canvas,xmin,ymin,w,h,img4]
    return canvasDict

def combine(c_img1, img1,optH,img2_chr_list,direction,blend):
    alpha = 0.8
    if not blend:
        if direction == 'right':
            canvas,xmin,ymin,w,h,img4 = img2_chr_list
            hinv = np.linalg.inv(optH)
            x,y = transformH([0,0], hinv, gscaleX, gscaleY)
            x = int(x)
            y = int(y)
            max_h = max(y,y - ymin) + max(h-(y - ymin) , img1.shape[0] - y)
            max_w = x + w
            tmp1 = np.zeros([max_h,max_w,3])
            tmp1[:img4.shape[0],-w:,:] = img4
            tmp1[abs(ymin): abs(ymin)+ img1.shape[0], 0: img1.shape[1],:] = c_img1
            tmp1 = tmp1.astype(int)
            posX = 0
            posY = abs(ymin)
            # plt.figure(figsize = [20,20])
            # plt.imshow(tmp1)
            return tmp1
        
        if direction == 'left':
            canvas,xmin,ymin,w,h,img4 = img2_chr_list
            hinv = np.linalg.inv(optH)
            x,y = transformH([0,0], hinv, gscaleX, gscaleY)
            x = int(x)
            y = int(y)
            max_h = abs(y) + max(h-abs(y) , img1.shape[0])
            max_w = abs(xmin) + img1.shape[1]
            tmp1 = np.zeros([max_h,max_w,3])
            tmp1[:img4.shape[0],:w] = img4
            tmp1[abs(ymin): abs(ymin)+ img1.shape[0], -img1.shape[1]:] = c_img1
            tmp1 = tmp1.astype(int)
            posX = max_w - img1.shape[1]
            posY = abs(ymin)
            # plt.figure(figsize = [20,20])
            # plt.imshow(tmp1)
            return tmp1
    if blend:
        if direction == 'right':
            canvas,xmin,ymin,w,h,img4 = img2_chr_list
            hinv = np.linalg.inv(optH)
            x,y = transformH([0,0], hinv, gscaleX, gscaleY)
            x = int(x)
            y = int(y)
            max_h = max(y,y - ymin) + max(h-(y - ymin) , img1.shape[0] - y)
            max_w = x + w
            tmp1 = np.zeros([max_h,max_w,3])
            tmp1[:img4.shape[0],-w:,:] = img4
            for i in range(abs(ymin),abs(ymin)+img1.shape[0]):
                for j in range(0,img1.shape[1]):
                    if np.sum(tmp1[i,j,:]) == 0:
                        tmp1[i, j,:] = c_img1[i - abs(ymin),j,:]
                    else:
                        tmp1[i, j,:] = alpha*c_img1[i - abs(ymin),j,:] + (1 - alpha)*tmp1[i,j,:]
            tmp1 = tmp1.astype(int)
            posX = 0
            posY = abs(ymin)
            # plt.figure(figsize = [20,20])
            # plt.imshow(tmp1)
            return tmp1
        
        if direction == 'left':
            canvas,xmin,ymin,w,h,img4 = img2_chr_list
            hinv = np.linalg.inv(optH)
            x,y = transformH([0,0], hinv, gscaleX, gscaleY)
            x = int(x)
            y = int(y)
            max_h = abs(y) + max(h-abs(y) , img1.shape[0])
            max_w = abs(xmin) + img1.shape[1]
            tmp1 = np.zeros([max_h,max_w,3])
            tmp1[:img4.shape[0],:w] = img4
            for i in range(abs(ymin),abs(ymin)+img1.shape[0]):
                for j in range(-img1.shape[1],0):
                    if np.sum(tmp1[i,j,:]) == 0:
                        tmp1[i, j,:] = c_img1[i - abs(ymin),j + img1.shape[1],:]
                    else:
                        tmp1[i, j,:] = alpha*c_img1[i - abs(ymin),j + img1.shape[1],:] + (1 - alpha)*tmp1[i,j,:]
            tmp1[abs(ymin): abs(ymin)+ img1.shape[0], -img1.shape[1]:] = c_img1
            tmp1 = tmp1.astype(int)
            posX = max_w - img1.shape[1]
            posY = abs(ymin)
            # plt.figure(figsize = [20,20])
            # plt.imshow(tmp1)
            return tmp1
imgs,keypoints,descriptors = readImages()
img1 = imgs[-2]
img2 = imgs[-1]
c_img1 = color_imgs[-2]
c_img2 = color_imgs[-1]
img2_chr_list,optH = get_specific_wrap(img1,img2,c_img1,c_img2)
img4 = img2_chr_list[-1].astype(int)
# plt.figure(figsize = [20,20])
# plt.imshow(img4)
tmp = combine(c_img1, img1, optH,img2_chr_list,'right',blend = 0)

img1 = imgs[-3]
img2  = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
c_img1 = color_imgs[-3]
c_img2 = tmp
img2_chr_list1,optH = get_specific_wrap(img1,img2,c_img1,c_img2)
img41 = img2_chr_list1[-1].astype(int)
# plt.figure(figsize = [20,20])
# plt.imshow(img41)
tmp1 = combine(c_img1, img1, optH,img2_chr_list1,'right',blend = 0)

img1 = imgs[1]
c_img1 = color_imgs[1]
img2  = imgs[0]
c_img2 = color_imgs[0]
img2_chr_list2,optH = get_specific_wrap(img1,img2,c_img1,c_img2)
img42 = img2_chr_list2[-1].astype(int)
# plt.figure(figsize = [20,20])
# plt.imshow(img42)
tmp2 = combine(c_img1,img1, optH,img2_chr_list2,'left',blend = 0)

img1 = imgs[2]
img2  = cv2.normalize(tmp2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
c_img1 = color_imgs[2]
c_img2 = tmp2
img2_chr_list3,optH = get_specific_wrap(img1,img2,c_img1,c_img2)
img43 = img2_chr_list3[-1].astype(int)
# plt.figure(figsize = [20,20])
# plt.imshow(img43)
tmp3 = combine(c_img1,img1, optH,img2_chr_list3,'left',blend = 0)


yf1 = abs(img2_chr_list1[2])
yf3 = abs(img2_chr_list3[2])

width = abs(img2_chr_list3[1]) + tmp1.shape[1]
height = max(yf3,yf1) + max(tmp3.shape[0] - yf3, tmp1.shape[0] - yf1)
final = np.zeros([height,width,3])
if yf3 >= yf1:
    final[:tmp3.shape[0],:tmp3.shape[1],:] = tmp3
    final[yf3 - yf1:(yf3 - yf1 + tmp1.shape[0]), -tmp1.shape[1]:,:] = tmp1

if yf1 > yf3:
    final[:tmp1.shape[0],-tmp1.shape[1]:,:] = tmp1
    final[yf1 - yf3:(yf1 - yf3 + tmp3.shape[0]), :tmp3.shape[1],:] = tmp3
    
final = final.astype(int)
plt.figure(figsize = [20,20])
plt.imshow(final)
plt.savefig('Mosaic.png')


# plt.figure(figsize = [20,20])
# plt.subplot(121),plt.imshow(tmp2),plt.title('Image1')
# plt.subplot(122),plt.imshow(img43),plt.title('Image1_Warped')
# plt.show()

# plt.figure(figsize = [20,20])
# plt.subplot(121),plt.imshow(color_imgs[2]),plt.title('Image2')
# plt.subplot(122),plt.imshow(tmp3),plt.title('Image 1-2 stitched')
# plt.show()

'''
hinv = np.linalg.inv(optH)
x,y = transformH([0,0], hinv, gscaleX, gscaleY)
xmin = img2_chr_list[1]
ymin = img2_chr_list[2]
plt.scatter(int(x) - xmin,int(y)-ymin,s = 500,c = 'red')


imgs,keypoints,descriptors = readImages()
Hs = getHs()
warpedImgs = get_all_warped(Hs)
img1 = imgs[-2]
img2_chr_list = warpedImgs[4]
optH = Hs[-1]
tmp = combine(img1,optH,img2_chr_list,'right')
optH = Hs[-2]
warpedImgsdict = get_wrap(optH,tmp)
img0 = imgs[-3]
tmp1 = combine(img0,optH,warpedImgsdict[0],'right')
'''

#matches,img1,img2,rows1,cols1,rows2,cols2,keypoints_1,keypoints_2 = returnMatches(0,1)
#img1cords,img2cords = getFeature_cords(matches,keypoints_1,keypoints_2)
#scaled_img1cords , scaled_img2cords = scaled(rows1, cols1, rows2, cols2, img1cords, img2cords)
#optH = RANSAC(matches)

"""
canvas,xmin,ymin,w,h = getCanvas(img2, optH)
img4 = Homographytransform_c(canvas,xmin,ymin,w,h)
#img4 = Homographytransform()

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)

i=50
plt.figure(figsize = [20,20])
plt.subplot(121),plt.imshow(img1,cmap = 'gray'),plt.title('IMG3')
plt.subplot(122),plt.imshow(img4,cmap = 'gray'),plt.title('IMG4_Transformed')
plt.show()
#cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
img5 = img4.copy()
img5[0:rows1,:cols1] = img1
plt.figure()
plt.imshow(img5, cmap = 'gray')
for i in range(0,len(matches[:50])):
    img3 = cv2.circle(img3,tuple(img1cords[i]),30,[0,0,255],-1)
    #img3 = cv2.circle(img3,(img2cords[i][0]+3024,img2cords[i][1]),30,[0,0,255],-1)
    x,y = transformH(scaled_img1cords[i],optH,cols2,rows2)
    print(x,y)
    img3 = cv2.circle(img3,(int(x+3024),int(y)),30,[0,0,255],-1)

# cv2.imshow("Result",img5)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

optH = Hs[3]
canvas,xmin,ymin,w,h,img4 = warpedImgs[4]
#img1 = imgs[2]
img1 = tmp
hinv = np.linalg.inv(optH)
x,y = transformH([0,0], hinv, cols2, rows2)
x = int(x)
y = int(y)
max_h = max(y,y - ymin) + max(h-(y - ymin) , img1.shape[0] - y)
max_w = x + w
tmp1 = np.zeros([max_h,max_w])
tmp1[:img4.shape[0],-w:] = img4
tmp1[abs(ymin)-Yoffset: abs(ymin)+ img1.shape[0]-Yoffset, 0: img1.shape[1]] = img1
plt.figure(figsize = [20,20])
plt.imshow(tmp1,cmap = 'gray')

"""
