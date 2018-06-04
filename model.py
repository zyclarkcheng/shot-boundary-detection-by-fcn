# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 21:02:59 2018

@author: clark
"""

#model for training
#input data: snippets of frames[batches , w, h, t, channels]
import tensorflow as tf
import numpy as np
def create_weights(name,shape):
    return tf.Variable(name,tf.truncated_normal_initializer(shape,std=0.1))
def create_bias(size):
    return tf.Variable(tf.constant(0.05,shape=[size]))
#def max_pool(name,input):
#    return tf.nn.max_pool3d(input,ksize=[])
    
temporal_size_conv1=3   
filter_size_conv1 = 5
num_channels_conv1=3 
num_filters_conv1 = 16
w1=create_weights('w1',[temporal_size_conv1,filter_size_conv1,filter_size_conv1,
                        num_channels_conv1,num_filters_conv1])
b1=create_bias('b1',num_filters_conv1)
s1=[1,2,2,1,1]

temporal_size_conv2=3   
filter_size_conv2 = 3
num_channels_conv2=num_filters_conv1 
num_filters_conv2 = 24
w2=create_weights('w2',[temporal_size_conv2,filter_size_conv2,filter_size_conv2,
                        num_channels_conv2,num_filters_conv2])
b2=create_bias('b2',num_filters_conv2)
s2=[1,2,2,1,1]


temporal_size_conv3=3   
filter_size_conv3 = 3
num_channels_conv3=num_filters_conv2 
num_filters_conv2 = 32
w3=create_weights('w3',[temporal_size_conv3,filter_size_conv3,filter_size_conv3,
                        num_channels_conv3,num_filters_conv3])
b3=create_bias('b3',num_filters_conv3)
s3=[1,2,2,1,1]

temporal_size_conv4=1   
filter_size_conv4 = 6
num_channels_con4=num_filters_conv3 
num_filters_conv4 = 12
w4=create_weights('w4',[temporal_size_conv4,filter_size_conv4,filter_size_conv4,
                        num_channels_conv4,num_filters_conv4])
b4=create_bias('b4',num_filters_conv4)
s4=[1,1,1,1,1]

temporal_size_conv5=4   
filter_size_conv5 = 1
num_channels_conv5=num_filters_conv4 
num_filters_conv5 = 2
w5=create_weights('w5',[temporal_size_conv5,filter_size_conv5,filter_size_conv5,
                        num_channels_conv5,num_filters_conv5])
b5=create_bias('b5',num_filters_conv5)
s5=[1,1,1,1,1]

image_size=64
depth=10
learning_rate=0.0001
train_from_scratch=1
training_epochs=50
image_path='./img_May_Oct.npy'
label_path='./labels_May_Oct.npy'
images=np.load(image_path)
labels=np.load(label_path)
len_files=np.shape(labels)[0]
batch_size=4

x = tf.placeholder(tf.float32, shape=[None,depth,image_size,image_size,num_channels], name='x')  
y = tf.placeholder(tf.float32, shape=[None, 2], name='y')

def create_conv_layer(input,w,b,s,operation_name):#NDWHC
    return tf.nn.conv3d(input,w,strides=s,padding='valid',data_format='NDHWC',operation_name)+b

layer1=create_conv_layer(x,w1,b1,s1,'conv1')
layer2=create_conv_layer(layer1,w2,b2,s2,'conv2')
layer3=create_conv_layer(layer2,w3,b3,s3,'conv3')
layer4=create_conv_layer(layer3,w4,b4,s4,'conv4')
layer5=create_conv_layer(layer4,w5,b5,s5,'conv5')#[batch_size 1 1 1 2]

logits=tf.reshape(layer5,[-1,2])

entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y)
loss=tf.reduce_mean(entropy,name='entropy_loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
saver = tf.train.Saver()


with tf.Session() as sess:
     if train_from_scratch:
    # Training cycle
        sess.run(tf.initialize_all_variables())
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_snippets = len_files-depth
            total_batch=int(total_snippets/batch_size)
            # Loop over all batches
            total_pred=np.zeros([0,2])
            for s in range(total_snippets-1):
                batch_x=np.zeros([0,depth,image_size,image_size,3])
                for i in range(total_batch-1):
                    batch_x=np.concatenate((batch_x,np.expand_dims(images[s:s+depth,:,:,:],axis=0)))
                #batch_size*temporal_size*img_size*img_size*channels        
                batch_x = images[batch_size,i]
                batch_y = labels[i*batch_size:(i+1)*batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
                pred_and_truth=l=np.concatenate((p,batch_y),axis=1)
                total_pred = np.concatenate((total_pred,pred_and_truth))
        # sample prediction
            label_value = batch_y          
            err = label_value-p
#            if is_learning_rate_decay:
#                print('learning rate:', sess.run(learning_rate))
#            else:
#                    print('learning rate:',learning_rate)
#            print ("num batch:", total_batch)
    
            # Display logs per epoch step
#            if epoch % display_step == 0:
#                print ("Epoch:", '%04d' % (epoch+1), "cost=", \
#                    "{:.9f}".format(avg_cost))
#                print ("[*]----------------------------")
#                for i in range(3):
#                    print ("label value:", label_value[i], \
#                        "estimated value:", p[i])
#                print ("[*]============================")
#                
#            if (epoch+1)%checkpoint_steps == 0:
#                save_path = saver.save(sess, "./checkpoint/model_"+str(epoch+1))
#                print("Model saved in path: %s" % save_path)
    

