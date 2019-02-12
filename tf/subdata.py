"""
Prepare training data for sub model for a specific class. 
"""

import numpy as np 
from PIL import Image 
from glob import glob 
import pickle


"""
Method to get augment non-defective images by selecting patches of smaller size. 
Arguments
imgsetx: Image list holding all the extracted non-defective images.
labelset: Label list holding the labels for the extracted images.
imgpath: Image path on device
gw: Original width of image
gh: Original height of image
stride: Stride for shifting patch on an image
nw: Extracted image width
nh: Extracted image height
"""
def getpatches_nondef(imgsetx,labelset,label,imgpath,gw,gh,stride,nw,nh):
  img = Image.open(imgpath)
  img = np.array(img,dtype=np.float32)
  img = np.divide(img,255.0)
  img = np.reshape(img,(gw,gh,1))
  ii = 0
  jj =0
  while ii < (gh-stride):
    jj = 0
    while jj < (gw-stride):
      image = img[ii:(ii+nh),jj:(jj+nw),:]
      imgsetx.append(image)
      labelset.append(label)
      jj = jj + stride
    ii = ii + stride

"""
Method to augment defective images by cropping patches using point lists as selected by user. 
Arguments
imgpath: Image Path on device
Label: Label for the image
imgsetx: Image list holding all the extracted defective images.
imgsety: Label list holding the labels for the extracted images.
pointset: Point list holding the points at which extraction needs to be done for the specific image. 
gw: Original width of image
gh: Original height of image
nw: Extracted image width
nh: Extracted image height
"""
def getrotations_def(imgpath,label,imgsetx,imgsety,pointset,gw,gh,nw,nh):
    image = Image.open(imgpath)
    image = np.array(image,dtype=np.float32)
    image = np.divide(image,255.0)
    image = np.reshape(image,(gw,gh,1))
    
    for point in pointset:
      xst = point[0]
      yst = point[1]
      img = image[yst:(yst+nh),xst:(xst+nw),:]
      #Augmenting the image by a factor of 8.
      imgsetx.append(img)
      imgsetx.append(np.fliplr(img))
      img = np.rot90(img,axes=(1,0))
      imgsetx.append(img)
      imgsetx.append(np.fliplr(img))
      img = np.rot90(img,axes=(1,0))
      imgsetx.append(img)
      imgsetx.append(np.fliplr(img))
      img = np.rot90(img,axes=(1,0))
      imgsetx.append(img)
      imgsetx.append(np.fliplr(img))
      for i in range(8):
        imgsety.append(label)
    

"""
Method to prepare the training and validations sets for a specific class.
Arguments
data_apth: Path where data is located on the device. 
class_no: Class number for which data sets are to be prepared. 
gw: Original width of Image
gh: Original height of Image
stride: Stride for shifting patch on an image
nw: Extracted image width
nh: Extracted image height
tr_range_nd: Number of non-defective images to be included in training set. 
tr_range_d: Number of defective images to be included in training set.
"""
def prep_imgs(data_path,class_no,gw,gh,nw,nh,stride,tr_range_nd,tr_range_d):
  tr_x = []
  tr_y = []
  val_x = []
  val_y = []
  ndef_label = 1
  def_label = 0
  print ('collecting images')
  i = class_no - 1 
  cl_nd = glob(data_path + ("Class%d//*" %(i+1)))         #non-defective images stored in Class(Class_no) folder, eg- Class1
  cl_d =  glob(data_path + ("Class%d_def//*" %(i+1)))     #defective images stored in Class(Class_no)_def folder, eg- Class1_def
  file_path = data_path + ("qcdlsub%dlist" %(i+1))        #extraction points for defective images are stored in qcdlsub(Class_no)list file as prepared by prepdatalist, eg- qcdlsub1list
  dbfile = open(file_path,"rb")
  db = pickle.load(dbfile)
  dbfile.close()
  perm_nd = db['ndefperm']        #get the permutation for the non-defective images
  perm_d = db['defperm']          #get the permutation for the defective images
  k = 0
  for k in range(len(perm_nd)):
       if k < tr_range_nd: 
          getpatches_nondef(tr_x,tr_y,ndef_label,cl_nd[perm_nd[k]],gw,gh,stride,nw,nh)
       
       else:
          getpatches_nondef(val_x,val_y,ndef_label,cl_nd[perm_nd[k]],gw,gh,stride,nw,nh)
  print ('got non-defective images of class %r length=%r' %(i+1,k))

  deftrp = db['deftrp%d'%(i+1)] #get point list for training images
  defvalp = db['defvalp%d'%(i+1)] #get point list for validation images
  k = 0  
  for k in range(len(perm_d)):
        if k < tr_range_d:
            getrotations_def(cl_d[perm_d[k]],def_label,tr_x,tr_y,deftrp[k],gw,gh,nw,nh)
                        
        else:
            getrotations_def(cl_d[perm_d[k]],def_label,val_x,val_y,defvalp[k-tr_range_d],gw,gh,nw,nh)
  print ("got defective images of class=%r length=%r" %(i+1,k))
    
  tr_x = np.array(tr_x)
  tr_y = np.array(tr_y)
  val_x = np.array(val_x)
  val_y = np.array(val_y)  
  print ("sub ",class_no,tr_x.shape,tr_y.shape,val_x.shape,val_y.shape)

  #shuffle training set
  permutation = np.random.permutation(tr_x.shape[0])
  tr_x = tr_x[permutation]
  tr_y = tr_y[permutation]
  return tr_x,tr_y,val_x,val_y
