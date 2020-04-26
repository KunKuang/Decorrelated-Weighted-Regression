# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:31:04 2019

@author: KKUN
"""

import tensorflow as tf
import numpy as np
import os

### Algorithm for Global Balancing Regression
def f_global_balancing_regression(train_or_test, X_in, Y_in, beta_true, beta_error_pre):
    
    if train_or_test == 1:
        learning_rate = 0.005; num_step = 5000; tol= 1e-8
        tf.reset_default_graph()
        Weight = f_global_balancing(X_in, learning_rate, num_step, tol)
        
        learning_rate = 0.001; num_steps = 3000; tol = 1e-8
        tf.reset_default_graph() 
        RMSE, beta_hat = f_weighted_regression(1, X_in, Y_in, Weight, learning_rate, num_steps, tol, beta_true, beta_error_pre)
        
        return RMSE, beta_hat, Weight
    
    else:
        tf.reset_default_graph()
        RMSE, beta_hat = f_weighted_regression(0, X_in, Y_in, np.ones([X_in.shape[0],1]), 0, 0, 0, 0, 0)
        
    return RMSE, beta_hat

def f_global_balancing(X_in, learning_rate, num_steps, tol):
    n,p = X_in.shape

    display_step = 1000

    X = tf.placeholder("float", [None, p])
    G = tf.Variable(tf.ones([n,1]))
    
    loss_balancing = tf.constant(0, tf.float32)
    for j in range(1,p+1):
        X_j = tf.slice(X, [j*n,0],[n,p])
        T = tf.slice(X, [0,j-1],[n,1])
        #balancing_j = tf.divide(tf.matmul(tf.transpose(G*G),tf.matmul(T,tf.cast(np.ones((1,p)),tf.float32))*X_j),tf.reduce_sum(G*G)) - tf.divide(tf.matmul(tf.cast(np.ones((1,n)),tf.float32),T),n)*tf.divide(tf.matmul(tf.cast(np.ones((1,n)),tf.float32),X_j),n)
        balancing_j = tf.divide(tf.matmul(tf.transpose(G*G),tf.matmul(T,tf.cast(np.ones((1,p)),tf.float32))*X_j),tf.constant(n, tf.float32)) - tf.divide(tf.matmul(tf.transpose(G*G),T),tf.reduce_sum(G*G))*tf.divide(tf.matmul(tf.transpose(G*G),X_j),tf.constant(n, tf.float32))
        loss_balancing += tf.norm(balancing_j,ord=2)
    
    loss_weight_sum = (tf.reduce_sum(G*G)-n)**2
    loss_weight_l2 = tf.reduce_sum((G*G)**2)
    
    loss = 2000.0/p*loss_balancing + 0.0005*loss_weight_sum + 0.00005*loss_weight_l2

    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    X_feed = X_in
    for j in range(p):
        X_j = np.copy(X_in)
        X_j[:,j] = 0
        X_feed = np.vstack((X_feed,X_j))
    
    l_pre = 0
    for i in range(1, num_steps+1):
        _, l, l_balancing, l_weight_sum, l_weight_l2 = sess.run([optimizer, loss, loss_balancing, loss_weight_sum, loss_weight_l2], feed_dict={X: X_feed})
        if abs(l-l_pre) <= tol:
            print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f ... %f' % (i, l, l_balancing, l_weight_sum, l_weight_l2))
            break
        l_pre = l
        if i % display_step == 0 or i == 1:
            print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f ... %f' % (i, l, l_balancing, l_weight_sum, l_weight_l2))
            
    W_final = sess.run(G*G)
    fw = open('weights_global_balancing.txt', 'w')
    for items in W_final:
        fw.write(''+str(items[0])+'\n')
    fw.close()
    
    Weight = sess.run([G*G])
    
    return  Weight[0]

def f_weighted_regression(train_or_test, X_in, Y_in, Weight_in, learning_rate, num_steps, tol, beta_true, beta_error_pre):
    n,p = X_in.shape

    display_step = 1000

    X = tf.placeholder("float", [None, p])
    Y = tf.placeholder("float", [None, 1])
    W = tf.placeholder("float", [None, 1])

    beta = tf.Variable(tf.random_normal([p, 1]))
    b = tf.Variable(tf.random_normal([1]))
    hypothesis = tf.matmul(X, beta) + b
    
    saver = tf.train.Saver()
    sess = tf.Session()
    
    if train_or_test == 1:
        loss_predictive = tf.divide(tf.reduce_sum(W*(Y-hypothesis)**2),tf.reduce_sum(W))
        loss_l1 = tf.reduce_sum(tf.abs(beta))
        
        loss =1*loss_predictive + 0.0*loss_l1
    
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
        
        sess.run(tf.global_variables_initializer())

        l_pre = 0
        for i in range(1, num_steps+1):
            _, l, l_predictive, l_l1 = sess.run([optimizer, loss, loss_predictive, loss_l1], feed_dict={X: X_in, W:Weight_in, Y:Y_in})
            if abs(l-l_pre) <= tol:
                print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f' % (i, l, l_predictive, l_l1))
                break
            l_pre = l
            if i % display_step == 0 or i == 1:
                print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f' % (i, l, l_predictive, l_l1))
                
        beta_estimated_error = np.sum(np.abs(sess.run(beta)-beta_true))
        if beta_estimated_error < beta_error_pre:
            if not os.path.isdir('models/f_weighted_regression/'):
                os.makedirs('models/f_weighted_regression/')
            saver.save(sess, 'models/f_weighted_regression/f_weighted_regression.ckpt')
        
        
    else:
        saver.restore(sess,'models/f_weighted_regression/f_weighted_regression.ckpt')
    
    RMSE = tf.sqrt(tf.reduce_mean((Y-hypothesis)**2))
    RMSE_error, beta_hat = sess.run([RMSE, beta], feed_dict={X: X_in, Y:Y_in})
    
    return  RMSE_error, beta_hat