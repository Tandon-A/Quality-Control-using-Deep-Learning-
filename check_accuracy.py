import numpy as np
import caffe
import matplotlib.pyplot as plt

#change mod_def to path of model definition 
mod_def = "D:\Abhishek_Tandon\sop_scm\model\deploy_qmain1_net.prototxt"
#change mod_weights to path of model weights
mod_weights ="D:\Abhishek_Tandon\sop_scm\model\snap\main\_iter_2000.caffemodel" 
#load Net defined by the model definition prototxt and the model weights
mod = caffe.Net(mod_def,mod_weights,caffe.TEST)

"""
mean value for the DAGM 2007 dataset. 
"""
mu = [118,118,118]
mu = np.array(mu)
transformer = caffe.io.Transformer({'data':mod.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))
transformer.set_mean('data',mu)
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0))
mod.blobs['data'].reshape(1,3,128,128)
acc = 0

#Open train.txt file 
f = open("D:\Abhishek_Tandon\sop_scm\main1_data.txt","r")
files = f.readlines()
for i in files:
    out = i.split(' ')
    img = caffe.io.load_image(out[0])
    tr_img = transformer.preprocess('data',img)
    mod.blobs['data'].data[...] = tr_img
    output = mod.forward()
    prob = output['prob'][0]
    print ("class is = %r label is = %r" %(prob.argmax(),int(out[1])))
    if int(out[1]) == prob.argmax():
        acc = acc+1


#print accuracy of model 
print (acc,len(files),float(acc/len(files)))
    

