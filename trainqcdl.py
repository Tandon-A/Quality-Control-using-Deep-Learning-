import tensorflow as tf
import numpy as np 
import os
import pickle
"""
Import qcdl class to build qcdl model.
Import subdata to prepare training and validation data for sub part.
"""
from qcdl import qcdl 
import subdata

"""
Method to shuffle dataset (training and validation) and produce batches.
Arguments
x: Image set
y: Label set corresponding to image set
batch_siz: size of the batches to be generated 
"""
def shuffle_batch(x,y,batch_size):
  perm = np.random.permutation(len(x))
  n_batches = len(x) // batch_size
  for batch_index in np.array_split(perm,n_batches):
    x_br, y_br = x[batch_index],y[batch_index]
    yield x_br, y_br 

"""
Method to train the qcdl model
Arguments
model: An object of qcdl class
model_dir: Path where model would be saved
batch_size: size of the batches to be generated
tr_x: Training image set
tr_y: Training label set corresponding to training image set
val_x: Validation image set
val_y: Validation label set corresponding to validation image set
lr_rate: Learning rate for the model
"""
def train(model,model_dir,batch_size,tr_x,tr_y,val_x,val_y,lr_rate):
  saver = tf.train.Saver()
  epoch = 0
  d = 0.012
  max_epoch = 300
  limit = 30                                #Training model with early stopping using a limit of 30 epochs without any improvement 
  perf = 10000
  max_perf = 10000
  print ("starting training")
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while epoch < max_epoch:
      lr_rate = lr_rate/(1+ d*epoch)        #learning rate policy
      
      #training
      for br_x,br_y in shuffle_batch(tr_x,tr_y,batch_size):
        sess.run(model.opt,feed_dict={model.inputs:br_x,model.outputs:br_y,model.is_training:True,model.lr_rate:lr_rate})        
         
      val_acc = []
      val_loss = []
      #validation 
      for brval_x,brval_y in shuffle_batch(val_x,val_y,batch_size):
        br_acc, br_loss = sess.run([model.acc,model.loss],feed_dict={model.inputs:brval_x,model.outputs:brval_y,model.is_training:False})
        val_acc.append(br_acc)
        val_loss.append(br_loss)    
      val_acc = np.mean(val_acc)
      val_loss = np.sum(val_loss)
      if perf - val_loss < 0.01:
        limit = limit - 1
      elif val_loss < max_perf:
        max_perf = val_loss
        perf = val_loss
        limit = 30
        print ("saving max performance model (lowest validation loss)")
        print ("model val_acc = %r val_loss = %r epoch = %r " %(val_acc,val_loss,epoch))
        saver.save(sess,model_dir,write_meta_graph=True)
      if limit == 0:
        print ("early stopping epoch=%r" %(epoch))
        break 
      epoch = epoch + 1
    print ('completed training, model saved')
    
    
    
def main(_): 
    if FLAGS.data_path is None or (not os.path.exists(FLAGS.data_path)):
        print ("Data Path doesn't exist")
    else:
        if not os.path.exists(FLAGS.model_dir):
            print ("creating model directory=%r" %(FLAGS.model_dir))
            os.makedirs(FLAGS.model_dir)
        
        if (FLAGS.model_part.lower()!= "global") and (FLAGS.model_part.lower()!="sub"):
          print ("Incorrect model part. Please mention either 'global' or 'sub' ('without quotes)")
        
        elif (FLAGS.model_part.lower()=="sub") and ((int(FLAGS.class_no) > 6) or (int(FLAGS.class_no) < 1)):
          print ("Incorrect class number. Please mention a class number amongst the possible classes (1-6)")
        
        else:
          input_shape = 128,128,1
          batch_size = 50
          lr_rate = 0.001
          if FLAGS.model_part.lower() == "global":
            out_classes = 6
            model_name = "global"
            dbfile_path = FLAGS.data_path + "/qcdlglobal"
            dbfile = open(dbfile_path,"rb")
            db = pickle.load(dbfile)
            tr_x = db['tr_x']
            tr_y = db['tr_y']
            val_x = db['val_x']
            val_y = db['val_y']
            print ("global ",tr_x.shape,tr_y.shape,val_x.shape,val_y.shape)
          else:
            out_classes = 2
            model_name = "sub" + FLAGS.class_no
            org_size = 512,512
            stride = 64
            tr_range_nd = 700
            tr_range_d = 106
            tr_x,tr_y,val_x,val_y = subdata.prep_imgs(FLAGS.data_path+"/",int(FLAGS.class_no),org_size[0],org_size[1],input_shape[0],input_shape[1],stride,tr_range_nd,tr_range_d)
          tf.reset_default_graph()
          model = qcdl(input_shape,out_classes,model_name)
          train(model, os.path.join(FLAGS.model_dir, model_name), batch_size, tr_x, tr_y, val_x, val_y, lr_rate)
          print ("training model= %r completed" %(model_name))
          
flags = tf.app.flags
flags.DEFINE_string("data_path",None,"Path of parent directory of class folders and point lists as prepared by using prepdatalist")
flags.DEFINE_string("model_part","global","Model to train, global or sub")
flags.DEFINE_string("class_no","1","Class number for which to train the sub part")
flags.DEFINE_string("model_dir","model_dir","Directory name to save checkpoints")
FLAGS = flags.FLAGS
    
if __name__ == '__main__':
    tf.app.run()
