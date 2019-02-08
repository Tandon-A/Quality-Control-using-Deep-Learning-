"""
Python Class file to generate the model architecture. 
"""

import tensorflow as tf 

reg_const = 0.00005         #regularization constant for l2 regularization


"""
A general method for a dense layer. 
Arguments
inputs_dense: Input to dense layer. 
num_neuron: Number of neurons in dense layer.. 
is_drop: Whether Dropout is to be used or not.
drop_prob: Keep probability of dropout.
is_bn: Whether Batch Normalization is to be used or not.
is_training: Mode of the model (training/test)
relu: Use of relu activation function. 
w_init: Kernel Initializer
kernel_reg: Kernel Regularizer
name: name of the operation
Returns output by defining a dense layer. 
"""
def general_dense(inputs_dense,num_neuron=1024,is_drop=False,drop_prob = 0.5,is_bn=False, is_training = False, relu=True, 
                w_init = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'),kernel_reg = tf.contrib.layers.l2_regularizer(reg_const) ,name='fc'):
  with tf.variable_scope(name):
    if is_drop == True:
      inputs_dense = tf.layers.dropout(inputs_dense,rate=drop_prob,training=is_training,name='dropout')
      
    hid = tf.layers.dense(inputs_dense,num_neuron,activation=None,kernel_initializer=w_init,kernel_regularizer=kernel_reg,name="fc")
    
    if is_bn == True:
      hid = tf.layers.batch_normalization(hid,training=is_training,name='bn')
    if relu == True:
      hid = tf.nn.relu(hid)
    return hid
 
"""
A general convolution method. 
Arguments
inputs_conv: Input to convolution operation. 
filters: Number of output filters. 
kernel: Kernel Size of each filter
stride: Stride to be used by filter. 
padding: 'SAME' or 'VALID'. 
relu: Use of relu activation function. 
is_bn: Whether Batch Normalization is to be used or not.
is_training: Mode of the model (training/test)
w_init: Kernel Initializer
b_init: Bias Initializer
kernel_reg: Kernel Regularizer
alpha: slope of leaky relu when input < 0
name: name of the operation
Returns the output of convolution operation. 
"""
def general_conv(inputs_conv,filters=64,kernel=3,stride=1,padding='VALID',relu=True, is_bn = False, is_training = False,
                 w_init = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'), b_init = tf.constant_initializer(0.0), 
                 kernel_reg = tf.contrib.layers.l2_regularizer(reg_const), alpha = 0.2,name='conv'):
  with tf.variable_scope(name):
    conv = tf.layers.conv2d(inputs_conv,filters,kernel,stride,padding,kernel_initializer=w_init,bias_initializer=b_init,kernel_regularizer=kernel_reg)
    
    if is_bn == True:
      conv = tf.layers.batch_normalization(conv,training=is_training,name='bn')
    
    if relu == True:
      conv = tf.nn.relu(conv,name='relu')
    else:
      conv = tf.nn.leaky_relu(conv,alpha=alpha,name='leaky_relu')
    return conv



"""
Class to define the qcdl model architecture.
"""
class qcdl:
  def __init__(self,inp_shape,out_classes,mod_name):
    self.inp_shape = inp_shape
    self.mod_name= mod_name
    self.out_classes = out_classes
    self.inputs, self.outputs,self.lr_rate,self.is_training = self.model_io(self.inp_shape)
    self.logits,self.pred = self.model_arc(self.inputs,self.is_training,self.out_classes)
    self.loss = self.model_loss(self.logits,self.outputs)
    self.opt = self.model_opt(self.loss,self.lr_rate)
    self.acc = self.model_acc(self.pred,self.outputs)
  
  """
  Method to create the placeholder values.
  """
  def model_io(self,inp_shape):
    inputs = tf.placeholder(dtype=tf.float32,shape=[None,inp_shape[0],inp_shape[1],inp_shape[2]],name='inputs')
    outputs = tf.placeholder(dtype=tf.int32,shape=None,name='output')
    lr_rate = tf.placeholder(dtype=tf.float32,shape=None,name='lr_rate')
    is_training = tf.placeholder_with_default(False,shape=None,name='is_training')
    return inputs,outputs,lr_rate,is_training
  
  """
  Method to define model architecture.
  """
  def model_arc(self,inputs,is_training,out_classes):
    with tf.variable_scope((self.mod_name + '/qcdlmodel')):
      c1 = general_conv(inputs,filters=32,kernel=3,stride=1,padding='SAME', relu=True, is_bn = True, is_training = is_training,name='c1')
      p1 = tf.nn.max_pool(c1,[1,2,2,1],[1,2,2,1],'SAME')
      c2 = general_conv(p1,filters=64,kernel=3,stride=1,padding='SAME',relu=True,name='c2')
      c3 = general_conv(c2,filters=64,kernel=3,stride=1,padding='SAME',relu=True, name='c3')
      p2 = tf.nn.max_pool(c3,[1,2,2,1],[1,2,2,1],'SAME')
      c4 = general_conv(p2,filters=128,kernel=3,stride=1,padding='SAME',relu=True, name='c4')
      c5 = general_conv(c4,filters=128,kernel=3,stride=1,padding='SAME',relu=True, name='c5')
      p3 = tf.nn.max_pool(c5,[1,2,2,1],[1,2,2,1],'SAME')
      p3_shape = p3.get_shape().as_list()
      p3_reshape = tf.reshape(p3,[-1,p3_shape[1]*p3_shape[2]*p3_shape[3]])
      fc1 = general_dense(p3_reshape,num_neuron=1024,is_drop=True,drop_prob=0.5,is_bn=True,is_training=is_training,relu=True,name='fc1')
      fc2 = general_dense(fc1,num_neuron=1024,is_drop=False,is_bn=True,is_training=is_training,relu=True,name='fc2')
      fc3 = general_dense(fc2,num_neuron=out_classes,is_drop=True,drop_prob=0.5,is_bn=False,is_training=is_training,relu=False,name='fc3')
      pred = tf.nn.softmax(fc3,name='pred')
      return fc3,pred
  
  """
  Method to calculate loss. Using cross entropy loss in combination with l2 regularization loss.
  """
  def model_loss(self,logits,labels):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)) + tf.losses.get_regularization_loss()
    return loss
  
  """
  Method to run optmization step.
  """
  def model_opt(self,loss,lr_rate):
    train_vars = tf.trainable_variables()
    train_vars = [var for var in train_vars if var.name.startswith((self.mod_name + '/qcdlmodel'))]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      #using Momentum Optimizer as staed in the paper. 
      model_opt = tf.train.MomentumOptimizer(lr_rate,momentum=0.9).minimize(loss,var_list=train_vars)
    return model_opt
  
  """
  Method to calculate accuracy.
  """
  def model_acc(self,pred,labels):
    return tf.reduce_mean(tf.cast(tf.nn.in_top_k(pred,labels,1),tf.float32))
