#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 05:56:24 2020

@author: sumendra
"""
##############################################
## Enabling GPU 
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
##############################################
## Disable Warning Messages
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
#################################################
## Library to Import
import tensorflow.compat.v1 as tf # if you are using tensorflow Version 2.x.x
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix                 
from tqdm import tqdm 
import cv2                   
tf.disable_v2_behavior() # if you are using Tensoflow Version 2.x.x

##############################################
## Manage GPU Memory utilization 
TF_ENABLE_GPU_GARBAGE_COLLECTION=False
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

##############################################
## Make changes based on your dataset 
learning_rate = 0.001
n_epochs = 50  
batch_size = 128
n_input = N # Image Dimension (Height X Width X color channel eg.100x100x3  N i.e 30000)
n_classes = C # Number of class C i.e 5 
dropout = 0.75 # Probability to keep units
dropout2 = 1.0 # Probability to keep units 
##############################################
## Creating Dataset from individual folder where each folder holds images for specific class
X=[]
Z=[]
IMG_SIZE=100 #Image Dimension e.g 100 x 100 

#CLASS1_DIR = 'path to directory'
#CLASS2_DIR = 'path to dirrectory'

def assign_label(img,class_type):
    return class_type

def make_train_data(class_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,class_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))
        
make_train_data('class_label1',CLASS1_DIR)
#print(len(X))
make_train_data('class_label2',CLASS2_DIR)
#print(len(X))

###################################################################
## Normalizing the dataset, label enconding and categorical based Onehot  
le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,5) # OneHot Coding n_lable interger value will be based on number of class lable
X=np.array(X)
X=X/255 # standard scaling the image pixel value i.e. 0~255 with in the range of 0~1

###################################################################
## Spliting Train and Test Dataset 
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

###################################################################
## print statsistics 
precision_scores_list = []
accuracy_scores_list = []

def print_stats_metrics(y_test, y_pred):    
    print('Accuracy: %.2f' % accuracy_score(y_test,   y_pred) )
    #Accuracy: 0.84
    accuracy_scores_list.append(accuracy_score(y_test,   y_pred) )
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print ("confusion matrix")
    print(confmat)
    print (pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    precision_scores_list.append(precision_score(y_true=y_test, y_pred=y_pred, average='weighted'))
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='weighted'))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred, average='weighted'))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred, average='weighted'))

#####################################################################
## Ploting Graph
def plot_metric_per_epoch():
    x_epochs = []
    y_epochs = [] 
    for i, val in enumerate(accuracy_scores_list):
        x_epochs.append(i)
        y_epochs.append(val)
    
    plt.scatter(x_epochs, y_epochs,s=50,c='lightgreen', marker='s', label='score')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('Score per epoch')
    plt.legend()
    plt.grid()
    plt.show()
    
########################################################################
##
def conv2d(x, W, b, strides=1):
    # Conv2D function, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


##########################################################################
##
def maxpool2d(x, k=2):
    # MaxPool2D function
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


################################################################
##
def layer(input, weight_shape, bias_shape):
    W = tf.Variable(tf.random_normal(weight_shape))
    b = tf.Variable(tf.random_normal(bias_shape))
    mapping = tf.matmul(input, W)   
    result = tf.add( mapping ,  b )
    return result


################################################################
##
def conv_layer(input, weight_shape, bias_shape):
    ##rr =raw_input()
    W = tf.Variable(tf.random.normal(weight_shape))
    b = tf.Variable(tf.random.normal(bias_shape))
    conv = conv2d(input, W, b)
    # Max Pooling (down-sampling)
    conv_max = maxpool2d(conv, k=2)
    return conv_max

################################################################
##
def fully_connected_layer(conv_input, fc_weight_shape, fc_bias_shape, dropout):   
    new_shape = [-1, tf.Variable(tf.random.normal(fc_weight_shape)).get_shape().as_list()[0]]
    fc = tf.reshape(conv_input, new_shape)
    mapping = tf.matmul(   fc, tf.Variable(tf.random.normal( fc_weight_shape))   )
    fc = tf.add( mapping, tf.Variable(tf.random.normal(fc_bias_shape))    )
    fc = tf.nn.relu(fc)
    # Apply Dropout
    fc = tf.nn.dropout(fc, dropout)
    return fc


###########################################################
## CNN Architecture 2 Hidden Layer Example: 16 and 36 filter with 1024 fully connected layer  
    
def inference_conv_net2(x, dropout):
    # Reshape input picture 
    # shape = [-1, size_image_x, size_image_y, 1 channel e.g. grey scale, 3 for RGB]
    x = tf.reshape(x, shape=[-1, 100, 100, 3])

    # Convolution Layer 1, filter 5x5 conv, 1 input, 16 outputs
    # max pool will reduce image from 28*28 to 14*14
    conv1 = conv_layer(x, [5, 5, 3, 16], [16] )
    
    # Convolution Layer 2, filter 5x5 conv, 16 inputs, 36 outputs
    # max pool will reduce image from 14*14 to 7*7
    conv2 = conv_layer(conv1, [5, 5, 16, 36], [36] )
    
    # Fully connected layer, 7*7*36 inputs, 128 outputs
    # Reshape conv2 output to fit fully connected layer input
    fc1 = fully_connected_layer(conv2, [25*25*36, 1024], [1024] , dropout)

    # Output, 128 inputs, 10 outputs (class prediction)
    output = layer(fc1 ,[1024, n_classes], [n_classes] )
    return output


###########################################################
## Loss Function
def loss_deep_conv_net(output, y_tf):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(output, y_tf)
    loss = tf.reduce_mean(xentropy) 
    return loss

###########################################################
## Cost Function
def training(cost):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost)
    return train_op

###########################################################
## Validation Function
def evaluate(output, y_tf):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_tf,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy

###########################################################
## Defining the placehoder for tensor to training dataset 
    
x_tf = tf.placeholder(tf.float32, shape=[None, 100,100,3])
y_tf = tf.placeholder(tf.float32, shape=[None,n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

###############################################################
## Inference Fuction to train and valid the output      
output = inference_conv_net2(x_tf, keep_prob) 
cost = loss_deep_conv_net(output, y_tf)

train_op = training(cost) 
eval_op = evaluate(output, y_tf)

##################################################################
## for metrics 
y_p_metrics = tf.argmax(output, 1)

##################################################################
# Initialize and run tensorflow session
 
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

##################################################################
## Batch Parameters

num_samples_train =  len(y_train)
num_batches = int(num_samples_train/batch_size)

###########################################################################################
## Main Loop
def main():
    for i in range(n_epochs):
        for batch_n in range(num_batches):
            sta= batch_n*batch_size
            end= sta + batch_size
    
    
            sess.run( train_op , feed_dict={x_tf: x_train[sta:end,:] , y_tf: y_train[sta:end, :], keep_prob: dropout       })
    
            loss, acc = sess.run([cost, eval_op], feed_dict={x_tf: x_train[sta:end,:] , y_tf: y_train[sta:end, :], keep_prob: dropout2})
    
    
            result = sess.run(eval_op, feed_dict={x_tf: x_test, y_tf: y_test, keep_prob: dropout2})
            result2, y_pred = sess.run([eval_op, y_p_metrics], feed_dict={x_tf: x_test, y_tf: y_test, keep_prob: dropout2})
    
    
            print ("test1 {},{}".format(i,result))
            print ("test2 {},{}".format(i,result2))
    
            y_true = np.argmax(y_test, 1)
            print (y_pred)
            print (y_true)
            print_stats_metrics(y_true, y_pred)
        if i == 49:
            plot_metric_per_epoch()

##########################################################################################

if __name__ == "__main__":
    main()
    
print ("<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>>>")

