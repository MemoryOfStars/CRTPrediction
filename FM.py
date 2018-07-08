# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 20:16:03 2018

@author: gmx
"""

#本应该是划分多个field分别进行DNN和FM的处理，这样可以处理一维特征和二维特征，也隐含了多维特征
#这里暂定划分为整数特征和分类特征两部分
#
#最后的caoncat_size应该为embedding_layer和deep_layer两部分
#

import tensorflow as tf
from sklearn.cross_validation import train_test_split
from dataInput import cate_feature,int_feature
import numpy as np


deep_layers = [13,39,26]                   #三层的DNN
k = len(cate_feature) + len(int_feature)   #Dimensions of 隐向量
embeddings_output_size = 39
#data load
#click_record = load_iris()    #Waiting to be Implemented
#x = click_record["data"]        #Features
#y = click_record["target"]      #Result
#Generate Embedding Layer
weights_mat = dict()

#embeddings
weights_mat["feature_embeddings"] = tf.Variable(tf.random_normal([len(cate_feature),k],0.0,0.01),name="feature_embeddings")
weights_mat["feature_bias"] = tf.Variable(tf.random_uniform([len(cate_feature),1],0.0,1.0),name="feature_bias")


#deep  layers
num_layer = len(deep_layers)
input_size = 13                       #input int_features
glorot = np.sqrt(2.0 / 13)
#layer_0
weights_mat["layer_0"] = tf.Variable(np.random.normal(loc = 0,scale = glorot,size = (input_size,deep_layers[0])),
                                       dtype=np.float32)
weights_mat["bias_0"] = tf.Variable(np.random.normal(loc = 0,scale = glorot,size=(1,deep_layers[0])),
                                     dtype=np.float32)
#layer_1
weights_mat["layer_1"] = tf.Variable(np.random.normal(loc = 0,scale = glorot,size = (deep_layers[0],deep_layers[1])),
                                       dtype=np.float32)
weights_mat["bias_1"] = tf.Variable(np.random.normal(loc = 0,scale = glorot,size=(1,deep_layers[1])),
                                     dtype=np.float32)
#layer_2
weights_mat["layer_2"] = tf.Variable(np.random.normal(loc = 0,scale = glorot,size = (deep_layers[1],deep_layers[2])),
                                       dtype=np.float32)
weights_mat["bias_2"] = tf.Variable(np.random.normal(loc = 0,scale = glorot,size=(1,deep_layers[2])),
                                     dtype=np.float32)


#final concat layer  ------concat   embeddings' outputs and deep layers' outputs
input_size = deep_layers[-1] + embeddings_output_size






x,y = x[y!=2],y[y!=2]
for i in range(len(y)):
    if y[i] == 0:
        y[i]=-1
        
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=9415)   #随即划分测试机和训练集
n = 4       #4个特征
k = 4       #辅助向量维度为4

#fm algorithm
w0 = tf.Variable(0.1)
w1 = tf.Variable(tf.truncated_normal([n]))
w2 = tf.Variable(tf.truncated_normal([n,k]))

x_ = tf.placeholder(tf.float32,[None,n])
y_ = tf.placeholder(tf.float32,[None])
batch = tf.placeholder(tf.int32)
                    #tile函数为在[batch,1]唯独下平铺w2
                    #batch = 70
w2_new = tf.reshape(tf.tile(w2,[batch,1]),[-1,4,k])
board_x = tf.reshape(tf.tile(x_,[1,k]),[-1,4,k])
board_x2 = tf.square(board_x)

q = tf.square(tf.reduce_sum(tf.multiply(w2_new,board_x),axis=1))
h = tf.reduce_sum(tf.multiply(tf.square(w2_new),board_x),axis=1)



y_fm = w0+tf.reduce_sum(tf.multiply(x_,w1),axis=1)+\
       1/2*tf.reduce_sum(q-h,axis=1)
       
cost = tf.reduce_sum(0.5*tf.square(y_fm - y_))
       

batch_fl = tf.cast(batch,tf.float32)
accury=(batch_fl+tf.reduce_sum(tf.sign(tf.multiply(y_fm,y_))))/(batch_fl*2)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_op,feed_dict={x_:x_train,y_:y_train,batch:70})
        print(sess.run(cost,feed_dict={x_:x_train,y_:y_train,batch:70}))
    print(sess.run(accury,feed_dict={x_:x_test,y_:y_test,batch:30}))

















