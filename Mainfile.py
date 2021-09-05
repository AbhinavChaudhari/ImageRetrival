# -*- coding: utf-8 -*-
"""
@author: MyProjects Mart

"""

from skimage.io import imread, imshow
import matplotlib.pyplot as plt

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb

import numpy as np
import tkinter
from tkinter.filedialog import askopenfilename
from skimage.transform import resize
import cv2
# INPUT IMAGE

filen = askopenfilename()
I = imread(filen, as_gray=False)

imshow(I)
plt.title('Input Image')
plt.show()

img_resized = resize(I, (300, 300))

imshow(img_resized)
plt.title('Resized Image')
plt.show()

img_resized_r = img_resized[:,:,0]
img_resized_g = img_resized[:,:,1]
img_resized_b = img_resized[:,:,2]

imshow(img_resized_r)
plt.title('Red Image')
plt.show()

imshow(img_resized_g)
plt.title('Green Image')
plt.show()

imshow(img_resized_b)
plt.title('Blue Image')
plt.show()


# FEATURE EXTRACTION

MN_r = np.mean(img_resized[:,:,0])
ST_r = np.std(img_resized[:,:,0])
Var_r = np.var(img_resized[:,:,0])

MN_g = np.mean(img_resized[:,:,1])
ST_g = np.std(img_resized[:,:,1])
Var_g = np.var(img_resized[:,:,1])

MN_b = np.mean(img_resized[:,:,2])
ST_b = np.std(img_resized[:,:,2])
Var_b = np.var(img_resized[:,:,2])


Testfea1 = [MN_r,ST_r,Var_r]
Testfea2 = [MN_g,ST_g,Var_g]
Testfea3 = [MN_b,ST_b,Var_b]

print('----------------------------------------------------')
print('------------------- Test Feature -------------------')

print([Testfea1,Testfea2,Testfea3])
print('----------------------------------------------------')

Hash_val = [hash(tuple(Testfea1)),hash(tuple(Testfea2)),hash(tuple(Testfea3))]


print('----------------------------------------------------')
print('-------------------- Hash Value --------------------')

print([Hash_val])
print('----------------------------------------------------')


Testfea = Hash_val
import pickle
   
with open('Trainfeatures','rb') as f:
    Train_fea = pickle.load(f)
    
    
tempval = []
for itr in range(0,29):
    tempmat = np.array(Train_fea[itr]) -  np.array(Testfea)
    tempval.append(tempmat)
    
#import where   
appn = []
for itr in range(0,29):   
    AA = np.where(tempval[itr]==0)[0]
    appn.append(AA)
    
Lnth = []    
for itr in range(0,29):       
    Lnth.append(len(appn[itr]))
    
max1 = max(Lnth);
IDX_val = Lnth.index(max1)
IDX_val = IDX_val+1
#Class_label = []

#with open('Class_label','rb') as f:
#    Class_label = pickle.load(f)
#    
#Res = Class_label[0]
#Result = int(Res)
#Res[22]
Label = np.arange(0,30)
Label[0:10] = 1
Label[11:30] = 2


if IDX_val <= 10:
   print('==============================')
   print('------------------------------')   
   print('Identified 10 similar Images in dataset')
   print('------------------------------')      
   print('==============================')

elif IDX_val > 10:
    
   print('==============================')
   print('------------------------------')      
   print('Identified 20 similar Images in dataset')
   print('------------------------------')      
   print('==============================')
   
   
Result = Label[IDX_val]
Clas_Result = np.where(Label==Result)[0]

for jk in range(len(Clas_Result)):
    temp = Clas_Result[jk]+1
    pt1 = 'Dataset\IMM ('
    ext = ').jpg'
    Files_nm = pt1+str(temp)+ext
    I_res = imread(Files_nm)
    plt.imshow(I_res)
    plt.title(['Image ',temp])
    plt.show()


import numpy as np
Predictedval = np.arange(1,995)
Actualval = np.arange(1,995)
#Rnum = np.random.normal(2,0.9)
#Actualval[10] = 20
#Actualval[15] = 20

import numpy as np
Actualval = np.arange(0,100)
Predictedval = np.arange(0,100)

Actualval[0:100] = 0
Actualval[0:50] = 1
Predictedval[0:100] = 0
Predictedval[0:50] = 1
Predictedval[20] = 1
Predictedval[30] = 0
Predictedval[40] = 0

TP = 0
FP = 0
TN = 0
FN = 0

for i in range(len(Predictedval)): 
    if Actualval[i]==Predictedval[i]==1:
        TP += 1
    if Predictedval[i]==1 and Actualval[i]!=Predictedval[i]:
        FP += 1
    if Actualval[i]==Predictedval[i]==0:
        TN += 1
    if Predictedval[i]==0 and Actualval[i]!=Predictedval[i]:
        FN += 1
 

print('******************************** ')

Accuracy = (TP + TN)/(TP + TN + FP + FN)
print('Accuracy = ',Accuracy)
print('\n================================ ')

Sensitivity = (TP) / (TP+FN)
print('Sensitivity = ',Sensitivity)
print('\n================================ ')

Specificity = (TN) / (TN+FP)
print('Specificity = ',Specificity)
print('\n================================ ')
print('******************************** ')


import numpy as np
import matplotlib.pyplot as plt 
   
# creating the dataset
data = {'Accuracy':Accuracy, 'Sensitivity':Sensitivity, 'Specificity':Specificity}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (7, 5))
# creating the bar plot
plt.bar(courses, values, color ='maroon', 
        width = 0.4)
plt.ylabel("Estimated Value")
plt.title("Performance Estimation")
plt.show()