# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 21:02:59 2018

@author: clark
"""

#model for training
#input data: snippets of frames[batches , w, h, t, channels]
import tensorflow as tf
import numpy as np
import os
def create_weights(varname,shape):
    return tf.get_variable(name=varname,shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1),dtype=tf.float32)
def create_bias(varname,size):
    return tf.get_variable(name=varname,shape=[size],initializer=tf.constant_initializer(0.05),dtype=tf.float32)
#def max_pool(name,input):
#    return tf.nn.max_pool3d(input,ksize=[])
    
temporal_size_conv1=3   
filter_size_conv1 = 5
num_channels_conv1=3 
num_filters_conv1 = 16
w1=create_weights('w1',[temporal_size_conv1,filter_size_conv1,filter_size_conv1,\
                        num_channels_conv1,num_filters_conv1])
b1=create_bias('b1',num_filters_conv1)
s1=[1,1,2,2,1]

temporal_size_conv2=3   
filter_size_conv2 = 3
num_channels_conv2=num_filters_conv1 
num_filters_conv2 = 24
w2=create_weights('w2',[temporal_size_conv2,filter_size_conv2,filter_size_conv2,\
                        num_channels_conv2,num_filters_conv2])
b2=create_bias('b2',num_filters_conv2)
s2=[1,1,2,2,1]


temporal_size_conv3=3   
filter_size_conv3 = 3
num_channels_conv3=num_filters_conv2 
num_filters_conv3 = 32
w3=create_weights('w3',[temporal_size_conv3,filter_size_conv3,filter_size_conv3,\
                        num_channels_conv3,num_filters_conv3])
b3=create_bias('b3',num_filters_conv3)
s3=[1,1,2,2,1]

temporal_size_conv4=1   
filter_size_conv4 = 6
num_channels_conv4=num_filters_conv3 
num_filters_conv4 = 16
w4=create_weights('w4',[temporal_size_conv4,filter_size_conv4,filter_size_conv4,\
                        num_channels_conv4,num_filters_conv4])
b4=create_bias('b4',num_filters_conv4)
s4=[1,1,1,1,1]

temporal_size_conv5=4   
filter_size_conv5 = 1
num_channels_conv5=num_filters_conv4 
num_filters_conv5 = 2
w5=create_weights('w5',[temporal_size_conv5,filter_size_conv5,filter_size_conv5,\
                        num_channels_conv5,num_filters_conv5])
b5=create_bias('b5',num_filters_conv5)
s5=[1,1,1,1,1]

image_size=64
depth=10
learning_rate=0.00001
train_from_scratch=1
training_epochs=500
image_path='./images808_64_64_3.npy'
label_path='./labels798_2.npy'
label1_path='./labels798_1.npy'
images=np.load(image_path)
labels=np.load(label_path)
label1=np.load(label1_path)
len_files=np.shape(images)[0]
batch_size=4
display_step=4
checkpoint_steps=20
num_class=2
with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, shape=[batch_size,depth,image_size,image_size,3], name='x')  
    y = tf.placeholder(tf.float32, shape=[batch_size, num_class], name='y')
    
    def create_conv_layer(varname,input,w,b,s):#NDWHC
        return tf.nn.conv3d(name=varname,input=input,filter=w,strides=s,padding='VALID')+b
    
    layer1=create_conv_layer('layer1',x,w1,b1,s1)
    layer2=create_conv_layer('layer2',layer1,w2,b2,s2)
    layer3=create_conv_layer('layer3',layer2,w3,b3,s3)
    layer4=create_conv_layer('layer4',layer3,w4,b4,s4)
    layer5=create_conv_layer('layer5',layer4,w5,b5,s5)#[batch_size 1 1 1 2]
    
    logits=tf.reshape(layer5,[-1,num_class],name='logits')
    pred=tf.reshape(tf.argmax(input=logits, axis=1,name='pred'),[logits.shape[0],1])
    
    entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y)
    cost=tf.reduce_mean(entropy,name='entropy_loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
saver = tf.train.Saver()



with tf.Session() as sess:
     if train_from_scratch:
    # Training cycle
        sess.run(tf.initialize_all_variables())
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_snippets = len_files-depth# 808-10=798
            total_batch=int(total_snippets/batch_size)#798/4=199
            total_pred=np.zeros([0,1])
            for s in range(total_batch-1):
                
                batch_x=np.zeros([0,depth,image_size,image_size,3])
                for i in range(batch_size):
                    batch_x=np.concatenate((batch_x,np.expand_dims(images[s:s+depth,:,:,:],axis=0)))
                #batch_x----[batch_size*temporal_size*img_size*img_size*channels ]  
                
                batch_y = labels[s*batch_size:(s+1)*batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})
                #shape of p is [4,1]               
                total_pred=np.concatenate((total_pred,p))
                avg_cost += c / total_batch
                

            if epoch % display_step == 0:
                writer=tf.summary.FileWriter('./log/',graph=tf.get_default_graph()) 
                writer.close()
                print ("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost))
            if (epoch)%checkpoint_steps == 0:
                save_path = saver.save(sess, "./checkpoint/model_"+str(epoch))
                print("Model saved in path: %s" % save_path)
        pred_and_truth=np.concatenate((total_pred,label1),axis=1)
        print('optimization finished')
     else :
            ckpt = tf.train.get_checkpoint_state('./checkpoint/')
            print ('Restoring from {}...'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
#            print(stem)
            restore_epochs = int(stem.split('_')[-1])
            
            for epoch in range(restore_epochs, restore_epochs+training_epochs):
                avg_cost = 0.
                total_snippets = len_files-depth# 808-10=798
                total_batch=int(total_snippets/batch_size)#798/4=199
                total_pred=np.zeros([0,1])
                for s in range(total_batch-1):
                    
                    batch_x=np.zeros([0,depth,image_size,image_size,3])
                    for i in range(batch_size):
                        batch_x=np.concatenate((batch_x,np.expand_dims(images[s:s+depth,:,:,:],axis=0)))
                    #batch_x----[batch_size*temporal_size*img_size*img_size*channels ]  
                    
                    batch_y = labels[s*batch_size:(s+1)*batch_size]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})
                    #shape of p is [4,1]               
                    total_pred=np.concatenate((total_pred,p))
                    avg_cost += c / total_batch       
                if epoch % display_step == 0:
                    writer=tf.summary.FileWriter('./log/',graph=tf.get_default_graph()) 
                    writer.close()
                    print ("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost))
                if (epoch)%checkpoint_steps == 0:
                    save_path = saver.save(sess, "./checkpoint/model_"+str(epoch))
                    print("Model saved in path: %s" % save_path)
            pred_and_truth=np.concatenate((total_pred,label1),axis=1)
            print('optimization finished')
        
#                
#            if (epoch+1)%checkpoint_steps == 0:
#                save_path = saver.save(sess, "./checkpoint/model_"+str(epoch+1))
#                print("Model saved in path: %s" % save_path)
    

