# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:47:33 2021

@author: Nilesh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
import cv2

from scipy.sparse import coo_matrix
from numpy.linalg import eig
from skimage.color import rgb2lab, lab2rgb
from scipy.sparse.linalg import inv

def color_diff(point1,point2,sigma,image):
    row1 = point1[0]
    col1 = point1[1]
    row2 = point2[0]
    col2 = point2[1]
    
    val1 = image[row1,col1,:]
    val2 = image[row2,col2,:]
    
    dR = int(val1[0]) - int(val2[0])
    dG = int(val1[1]) - int(val2[1])
    dB = int(val1[2]) - int(val2[2])
    
    diff = (abs(dR) + abs(dG) + abs(dB))/3.0
    
    return diff/(sigma**2)

def feature_similartiry(point1, point2,sigma,image):
    row1 = point1[0]
    col1 = point1[1]
    row2 = point2[0]
    col2 = point2[1]
    
    val1 = rgb2lab(image[row1,col1,:]) * 10e5
    val1 = val1[0] * val1[1] * val1[2] 
    
    val2 = rgb2lab(image[row2,col2,:]) * 10e5
    val2 = val2[0] * val2[1] * val2[2] 
    
    return abs(val1 - val2)/(sigma**2)

def feature_similartiry_gray(point1, point2,sigma,image):
    row1 = point1[0]
    col1 = point1[1]
    row2 = point2[0]
    col2 = point2[1]
    
    val1 = image[row1,col1]
    val2 = image[row2,col2]
    
    return abs(val1 - val2)/(sigma**2)
    
def dist_similarity(point1, point2,sigma):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return d/(sigma**2)

def weight(point1, point2,sigma1, sigma2, image):
    #print(point1,point2)
    d = dist_similarity(point1, point2,sigma1)
    #f = feature_similartiry(point1, point2, sigma2, image)
    f = color_diff(point1, point2, sigma2, image)
    return(np.exp(-d - f))

def add_clusters(img):
    img[0:10,0:10,:] = [150,10,10]
    img[10:20,10:20,:] = [10,10,150]
    #img[15:20,15:20,:] = [10,150,10]
    
    return img.astype('int')

def c2p(cords,w,h):
    if cords[0] < 0 or cords[0]>= h:
        return -1
    if cords[1] < 0 or cords[1] >= w:
        return -1
    point = cords[0] + cords[1]*h
    return point

def p2c(point,w,h):
    cords= [0,0]
    cords[0] = point % h
    cords[1] = int(point / h)
    return cords

def get_neighbours(cords,w,h,Points):
    n = []
    for i in range(-1,2):
        for j in range(-1,2):
            if i == 0 and j == 0:
                continue
            row = cords[0] + i
            col = cords[1] + j
            
            if row < 0 or row >= w:
                continue
            
            if col < 0 or col >= h:
                continue
            
            n.append([row,col])
    return n

def get_neighbours_m(cords,Points):
    n = []
    for i in range(-1,2):
        for j in range(-1,2):
            if i == 0 and j == 0:
                continue
            row = cords[0] + i
            col = cords[1] + j
            
            n_pt = c2p([row,col],w,h)
            if n_pt in Points:
                n.append([row,col])
    return n

def wt_matrix(img):
    w = img.shape[0]
    h = img.shape[1]
    W = np.zeros([w * h, w * h])
    V = np.zeros([w * h, w * h])
    
    for col in range(0,w):
        for row in range(0,h):
            cords = [row,col]
            point1 = c2p(cords,w,h)
            neighbours = get_neighbours(cords,w,h)
            for neighbr in neighbours:
                w_ij = weight(cords, neighbr, 6,6,img)
                point2 = c2p(neighbr,w,h)
                W[point1,point2] = w_ij
                
    return(W)


def create_map_dict(Points):
    p2indx = {}
    indx2p = {}
    i = 0
    for point in Points:
        p2indx[point] = i
        indx2p[i] = point
        
        i += 1
        
    return p2indx,indx2p


def m_wt_matrix(Points,Img):
    n = len(Points)
    #W = np.zeros([n,n])
    
    p2indx,indx2p = create_map_dict(Points)
    rows = []
    cols = []
    data = []
    for point in Points:
        cords = p2c(point,w,h)
        neighbours = get_neighbours_m(cords,Points)
        #print(cords, neighbours)
        for neighbr in neighbours:
            w_ij = weight(cords, neighbr,1,1,Img)
            point2 = c2p(neighbr,w,h)
            i = p2indx[point]
            j = p2indx[point2]
            rows.append(i)
            cols.append(j)
            data.append(np.round(w_ij, 2))
        
    W = coo_matrix((data,(rows,cols)), shape = [n,n])
            
            
    return W, p2indx, indx2p

# Note this function return directly D^-1/2 matrix
def D_matrix(W):
    n = W.get_shape()[0]
    diag_V = np.sum(W,axis = 1)
    #diag_V = np.sqrt(diag_V)
    diag_V = np.array(diag_V).reshape((-1,)) + 0.001
    #diag_V = 1/diag_V
    indices = list(np.arange(0,n))
    D = coo_matrix((diag_V,(indices,indices)), shape = [n,n])
    #D = inv(D)
    #D = np.diag(diag_V)
    return D

def getClusters(Points_list, img, indx2p = None):
    global class_color
    for points in Points_list:
        for point in points:
            [row,col] = p2c(point,w,h)
            img[row,col] = class_color
        
        class_color += 1
    return img

def extract_classes(eigV):
    global CLASS_COUNTER
    class_dict = {}
    for val in eigV:
        if val not in class_dict:
            class_dict[val] = [CLASS_COUNTER,0]
            CLASS_COUNTER += 1
        
        else:
            class_dict[val][1] += 1
            
    print(class_dict)
    return(class_dict)

def get_Points(eigV, p2indx = {}, indx2p = {}):
    
    class_dict = extract_classes(eigV)
    indxlist = []
    n_indxlist = []
    
    for u_val in class_dict:
        indx = np.where(eigV == u_val)[0]
        indxlist.append(indx)
        class_dict[u_val].append(indx)
        
    if len(p2indx):
        for n_indx in indxlist:
            tmp = []
            for i in n_indx:
                tmp.append(indx2p[i])
            n_indxlist.append(tmp)
            
        return(n_indxlist,class_dict)
    
    else:
        return(indxlist,class_dict)
    

CLASS_COUNTER = 1
class_color = 0
#MASK_IMG = np.zeros([w,h])
#image = cv2.imread('test.jpg')
#w = image.shape[1]
#h= image.shape[0]

#mask_img = np.zeros([w,h])
image = np.zeros([20,20,3]).astype('int')
image = cv2.imread('redhat.jpg')
plt.figure(3)
plt.imshow(image)
aspect_ratio = image.shape[1]/image.shape[0]
size = [int(aspect_ratio*150),150]
image = cv2.resize(image,tuple(size),interpolation = cv2.INTER_AREA)
h,w = image.shape[0],image.shape[1]
mask_img = np.zeros([h,w])
#image = np.random.normal(50,4, size = [w,h,3]).astype('int')

#mask_img = np.zeros([w,h])
#image = np.zeros([w,h]).astype('int')

#image = add_clusters(image)
plt.figure(1)
plt.imshow(image)
eigV = np.ones(w*h)

p2indx = {}
indx2p = {}
img_counter = 0

import time
global mask_img_updates
mask_img_updates = 0
max_classes = 2
depth_per_branch = {0:3, 1:3, 2:2}
depth_per_branch_track = {0:0, 1:0, 2:0}

def get_num_points(PList):
    nL = []
    for i,P in enumerate(PList):
        nL.append(len(P))
    return(nL)


def norm_cut(Points, p2indx,indx2p,parent,flag):
    global mask_img, mask_img_updates
    
    W, p2indx, indx2p = m_wt_matrix(Points,image)
        
    D = D_matrix(W)
    
    #A = D_half.dot(W)
    #A = A.dot(D_half)
    
    eig_vals , eig_vec = eigs(D - W,M = D,k = 2, which = 'SM')
    #eig_vals , eig_vec = eig(A)
    eig_vec = np.round(eig_vec,3).astype('float')
    
    eigV = eig_vec[:,1]
    print("Eigen Values: ",eig_vals[1])
    mask_img = getClusters([Points],mask_img, indx2p)
    depth_per_branch_track[parent] += 1
    Points_list,class_dict = get_Points(eigV, p2indx, indx2p)
    #if len(Points_list) > 3 or len(Points_list) == 1:
        #return 0
    Points_list = sorted(Points_list, key = len, reverse = True)
    
    if depth_per_branch[parent] < depth_per_branch_track[parent]:
        return 0
    
    nL= get_num_points(Points_list)
    for i,Points in enumerate(Points_list):
        percent_accounted = len(Points)/sum(nL)
        print("Percent ", percent_accounted, parent)
        print(nL)
        if percent_accounted < 0.2:
            break
        if flag:
            parent = i+1
            depth_per_branch[parent] = 1
        if len(Points)> 20:
            norm_cut(Points, p2indx, indx2p, parent,0)
            
    '''
    for i,class_val in enumerate(class_dict):
        Points = class_dict[class_val][-1]
        if len(Points) > 20:
            norm_cut(Points,p2indx,indx2p)
            
    '''
"""
def norm_cut(Points, p2indx,indx2p):
    global mask_img
    W, p2indx, indx2p = m_wt_matrix(Points,image)
    D = D_matrix(W)
    eig_vals , eig_vec = eigs(D - W,M = D,k = 2, which = 'SM')
    eig_vec = np.round(eig_vec,3).astype('float')
    eigV = eig_vec[:,1]

    print("Eigen Values: ",eig_vals[1])
    mask_img = getClusters([Points],mask_img, indx2p)
    Points_list,class_dict = get_Points(eigV, p2indx, indx2p)
    if len(Points_list) > 3 or len(Points_list) == 1:
        return 0
    for Points in Points_list:
        if len(Points):
            norm_cut(Points,p2indx,indx2p)
"""

Points_list,class_dict = get_Points(eigV, p2indx, indx2p)
norm_cut(Points_list[0], p2indx, indx2p,0,1)
plt.figure(2)
plt.imshow(mask_img)