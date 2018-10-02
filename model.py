import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import utils2
import gc

#Define training and validation size

n_training = 717160
n_validation = 96000
n_test = 892816
 

class Model():
    def __init__(self):
        self.lrate = 0.005
        self.batch_size = 128 
        self.ntrain = 717160
        self.nval = 96000
        self.ntest = 892816
        self.nclasses = 2
        self.training =True
        self.keep_prob = tf.constant(0.90)
        self.hidden_size = 128

    def get_data(self,prep_train_data,target,prep_test_data):
        with tf.name_scope('data'):
            # Create dataset and iterator
            train,val,test = utils2.get_train_test_data(prep_train_data,target,prep_test_data,self.ntrain,self.nval,self.ntest)

            train_data = tf.data.Dataset.from_tensor_slices(train)
            train_data = train_data.shuffle(100000)
            train_data = train_data.batch(self.batch_size)

            val_data = tf.data.Dataset.from_tensor_slices(val)
            val_data = val_data.batch(self.batch_size)


            test_data = tf.data.Dataset.from_tensor_slices(test)
            test_data = test_data.batch(self.batch_size)

            iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                    train_data.output_shapes)

            X , label = iterator.get_next()
            _,self.m = X.shape
            self.X = tf.cast(X,dtype=tf.float32)
            self.label = tf.one_hot(label,self.nclasses)

            self.train_init = iterator.make_initializer(train_data)
            self.val_init = iterator.make_initializer(val_data)
            self.test_init = iterator.make_initializer(test_data)
      

    def inference(self):
        with tf.variable_scope('logreg',reuse=tf.AUTO_REUSE) as scope:
            # Create weights and bias
            # w is initialized to random variables with mean 0 and stddev 0.01 
            # b is initialized to zero
            w = tf.get_variable(name='weights',dtype=tf.float32,shape=[32,self.nclasses],
                              initializer = tf.random_normal_initializer(0 , 0.01))
            b = tf.get_variable(name='bias',shape=[self.nclasses],
                             initializer = tf.zeros_initializer())

            # build model 
            # the model that returns logits
            layer1 = tf.layers.dense(self.X,128,activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0,0.01),
                                   name='dense_layer1')
            #dropout1 = tf.nn.dropout(layer1,self.keep_prob,name='dropout1')

            layer2 = tf.layers.dense(layer1,64,activation=tf.nn.relu,
                                   kernel_initializer = tf.random_normal_initializer(0,0.01),
                                   name='dense_layer2')
            #dropout2 = tf.nn.dropout(layer2,self.keep_prob,name='dropout2')

            layer3 = tf.layers.dense(layer2,32,activation=tf.nn.relu,
                                   kernel_initializer = tf.random_normal_initializer(0,0.01),
                                   name='dense_layer3')

            self.logits = tf.matmul(layer3,w) + b

    def create_loss(self):
        with tf.name_scope('loss'):
            # define loss function
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits , labels = self.label)
            self.loss = tf.reduce_mean(entropy,name='loss')
      
    def create_optimizer(self):
        with tf.name_scope('optimizer'):      
            # define training op 
            self.optimizer = tf.train.AdamOptimizer(self.lrate).minimize(self.loss)

    def eval_model(self):
        with tf.name_scope('eval'):      
            # calculate accuracy 
            self.preds = tf.nn.softmax(self.logits)
            self.predicted_labels = tf.argmax(self.preds,1)
            correct_preds = tf.equal(tf.argmax(self.preds,1),tf.argmax(self.label,1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds,dtype=tf.float32))

    def build_model(self,prep_train_data,target,prep_test_data):
        self.get_data(prep_train_data,target,prep_test_data)
        self.inference()
        self.create_loss()
        self.create_optimizer()
        self.eval_model()
    
    def train(self,n_epochs):
        # start training loop 
        init = tf.global_variables_initializer()
        
        utils2.safe_mkdir('checkpoints')
        utils2.safe_mkdir('checkpoints/model2')
        
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(n_epochs):
                sess.run(self.train_init)
                total_loss = 0.0
                n_batches = 0
                step = 0
                train_acc = 0

                try:
                    while True:
                        _,batch_loss,acc = sess.run([self.optimizer,self.loss,self.accuracy])
                        total_loss += batch_loss
                        train_acc += acc
                        n_batches += 1
                        step += 1
                        if step%100 == 0:
                            print('Step {} : Loss {}'.format(step,batch_loss))
                except tf.errors.OutOfRangeError:
                    pass
                print('Average loss at epoch {} is {}'.format(epoch,total_loss/n_batches))
                print('Average train accuracy at epoch {} is {}'.format(epoch,train_acc/self.ntrain))

            
            #calculate validation set accuracy
            sess.run(self.val_init)
            total_acc = 0
            val_pred = []
            try:
                while True:
                    acc,pred = sess.run([self.accuracy,self.predicted_labels])
                    total_acc += acc
                    val_pred.append(pred)
            except tf.errors.OutOfRangeError:
                pass
            print('Average Validation accuracy {}'.format(total_acc / 96000))
      
            #calculate test set predictions
            test_pred = []

            sess.run(self.test_init)
            self.training = False
            try:
                while True:
                    pred = sess.run(self.predicted_labels)
                    test_pred.append(pred)
            except tf.errors.OutOfRangeError:
                pass
        pass
        return val_pred,test_pred


if __name__ == '__main__':
   
    train_path = './ds_data/data_train.csv'
    test_path = './ds_data/data_test.csv'
    
    prep_train_data,prep_test_data,target,test_id = utils2.get_preprocessed_data(train_path,test_path)
    print(prep_train_data.shape)
    print(prep_test_data.shape)
    print(target.shape)
    
    gc.collect()
    #Build Model
    model = Model()
    model.build_model(prep_train_data,target,prep_test_data)
    predictions = model.train(20)
    _,test_pred = predictions
    print(type(test_pred))
    
    test_pred = np.concatenate(test_pred,axis=0)
    test_pred = test_pred.astype(int)

    test_pred = np.column_stack((test_id[:892816],test_pred))
    
    #write to csv file
    with open('test.csv','a+') as f:
        np.savetxt(f,test_pred,fmt = '%d',delimiter=",")


