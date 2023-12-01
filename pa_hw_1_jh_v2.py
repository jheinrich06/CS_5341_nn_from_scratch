#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:30:54 2023

@author: Jason Heinrich
"""
import math
import random
import numpy as np
import matplotlib.pyplot as plt


def trainModel(error_rate, activation_mode, random_mode, ignore_bias, label ):
    #Initial weights and inputs
   
    w1 = 0.1
    w2 = 0.2
    w3 = 0.25
    w4 = 0.3
    w5 = 0.8
    w6 = 0.75
    w7 = 0.7
    w8 = 0.8
    w9 = 0.6
    w10 = 0.65
    w11 = 0.8
    w12 = 0.2
    
    if ignore_bias == True:
        w9 = 0
        w10 = 0
        w11 = 0
        w12 = 0
    
    if random_mode == True:
        w1 = random.random()
        w2 = random.random()
        w3 = random.random()
        w4 = random.random()
        w5 = random.random()
        w6 = random.random()
        w7 = random.random()
        w8 = random.random()
        w9 = random.random()
        w10 = random.random()
        w11 = random.random()
        w12 = random.random()
        
    w_start_1 = w1
    w_start_2 = w2
    w_start_3 = w3
    w_start_4 = w4
    w_start_5 = w5
    w_start_6 = w6
    w_start_7 = w7
    w_start_8 = w8
    w_start_9 = w9
    w_start_10 = w10
    w_start_11 = w11
    w_start_12 = w12
    
    start_arr = [w_start_1, w_start_2, w_start_3, w_start_4, w_start_5, w_start_6, w_start_7, w_start_8, w_start_9, w_start_10, w_start_11, w_start_12]
    
    i1 = 0.45
    i2 = 0.1
    
    #Targets and learning rate
    target_o1 = 0.99
    target_o2 = 0.01
    
    l_rate = 0.08
    
    #placeholder for outputs
    net_h1 = 0
    net_h2 = 0
    out_h1 = 0
    out_h2 = 0
    net_o1 = 0
    net_o2 = 0
    out_o1 = 0
    out_o2 = 0
    e_o1 = 0
    e_o2 = 0
    e_total = 1
    
    #Utility calculations placeholders for partial derivative chains
    out_o1_minus_target_o1 = 0
    out_o2_minus_target_o2 = 0
    out_o1_times_1_minus_o1 = 0
    out_o2_times_1_minus_o2 = 0
    out_h1_times_1_minus_h1 = 0
    out_h2_times_1_minus_h2 = 0
    
    out_o1_paired = 0
    out_o2_paired = 0
    
    #Partial derivative placeholders
    pd_w1 = 0
    pd_w2 = 0
    pd_w3 = 0
    pd_w4 = 0
    pd_w5 = 0
    pd_w6 = 0
    pd_w7 = 0
    pd_w8 = 0
    pd_w9 = 0
    pd_w10 = 0
    pd_w11 = 0
    pd_w12 = 0
    
    #Updated weights placeholders
    new_w1 = 0
    new_w2 = 0
    new_w3 = 0
    new_w4 = 0
    new_w5 = 0
    new_w6 = 0
    new_w7 = 0
    new_w8 = 0
    new_w9 = 0
    new_w10 = 0
    new_w11 = 0
    new_w12 = 0
    
    #Iteration counter
    iterations_run = 0
    euclidean_distance = 0
    
    
    
    while e_total > error_rate:
        #Get outputs for current epoch
        if ignore_bias == True:
            net_h1 = (w1*i1) + (w4*i2)
            net_h2 = (w3*i1) + (w2*i2)
        else:
            net_h1 = (w1*i1) + (w4*i2) + (w9*1)
            net_h2 = (w3*i1) + (w2*i2) + (w10*1)
        
        if activation_mode == 'relu':
            out_h1 = max(0, net_h1)
            out_h2 = max(0, net_h2)
        else:
            out_h1 = 1/(1+math.exp(-net_h1))
            out_h2= 1/(1+math.exp(-net_h2))
        
        if ignore_bias == True:
            net_o1 = (w5*out_h1) + (w7*out_h2)
            net_o2 = (w6*out_h1) + (w8*out_h2)
        else:
            net_o1 = (w5*out_h1) + (w7*out_h2) + (w11*1)
            net_o2 = (w6*out_h1) + (w8*out_h2) + (w12*1)
        

        out_o1 = 1/(1+math.exp(-net_o1))
        out_o2= 1/(1+math.exp(-net_o2))
        
        e_o1 = (1/2)*(target_o1 - out_o1)*(target_o1 - out_o1)
        e_o2 = (1/2)*(target_o2 - out_o2)*(target_o2 - out_o2)
        
        e_total = e_o1 +e_o2
        
        # Get utility calculations for partial derivative construction
        out_o1_minus_target_o1 = out_o1 - target_o1
        out_o2_minus_target_o2 = out_o2 - target_o2
        

        pd_relu_o1 = out_o1*(1-out_o1)
        pd_relu_o2 = out_o2*(1-out_o2)

            
        out_o1_times_1_minus_o1 = out_o1*(1-out_o1)
        out_o2_times_1_minus_o2 = out_o2*(1-out_o2)
        out_h1_times_1_minus_h1 = out_h1*(1-out_h1)
        out_h2_times_1_minus_h2 = out_h2*(1-out_h2)
        
        if net_h1 > 0:
            pd_relu_h1 = 1
        else:
            pd_relu_h1 = 0
            
        if net_h2 > 0:
            pd_relu_h2 = 1
        else:
            pd_relu_h2 = 0
        
        
        out_o1_paired = out_o1_minus_target_o1 * out_o1_times_1_minus_o1
        out_o2_paired = out_o2_minus_target_o2 * out_o2_times_1_minus_o2
        out_o1_paired_relu = out_o1_minus_target_o1 * pd_relu_o1
        out_o2_paired_relu = out_o2_minus_target_o2 * pd_relu_o2
        
        #Calculaate partial derivatives
        if activation_mode == 'relu':
            pd_w1 = (pd_relu_h1*i1)*((out_o1_paired_relu*w5)+(out_o2_paired_relu*w6))
            pd_w2 = (pd_relu_h2*i2)*((out_o1_paired_relu*w7)+(out_o2_paired_relu*w8))
            pd_w3 = (pd_relu_h2*i1)*((out_o1_paired_relu*w7)+(out_o2_paired_relu*w8))
            pd_w4 = (pd_relu_h1*i2)*((out_o1_paired_relu*w5)+(out_o2_paired_relu*w6))
            pd_w5 = out_o1_paired_relu*out_h1
            pd_w6 = out_o2_paired_relu*out_h1
            pd_w7 = out_o1_paired_relu*out_h2
            pd_w8 = out_o2_paired_relu*out_h2
            pd_w9 = (w5*out_o1_paired_relu*pd_relu_h1) +(w6*out_o2_paired_relu*pd_relu_h1)
            pd_w10 = (w7*out_o1_paired_relu*pd_relu_h2) +(w8*out_o2_paired_relu*pd_relu_h2)
            pd_w11 = out_o1_paired_relu
            pd_w12 = out_o2_paired_relu
        
        else:
            pd_w1 = (out_h1_times_1_minus_h1*i1)*((out_o1_paired*w5)+(out_o2_paired*w6))
            pd_w2 = (out_h2_times_1_minus_h2*i2)*((out_o1_paired*w7)+(out_o2_paired*w8))
            pd_w3 = (out_h2_times_1_minus_h2*i1)*((out_o1_paired*w7)+(out_o2_paired*w8))
            pd_w4 = (out_h1_times_1_minus_h1*i2)*((out_o1_paired*w5)+(out_o2_paired*w6))
            pd_w5 = out_o1_paired*out_h1
            pd_w6 = out_o2_paired*out_h1
            pd_w7 = out_o1_paired*out_h2
            pd_w8 = out_o2_paired*out_h2
            pd_w9 = (w5*out_o1_paired*out_h1_times_1_minus_h1) +(w6*out_o2_paired*out_h1_times_1_minus_h1)
            pd_w10 = (w7*out_o1_paired*out_h2_times_1_minus_h2) +(w8*out_o2_paired*out_h2_times_1_minus_h2)
            pd_w11 = out_o1_paired
            pd_w12 = out_o2_paired
        
        #Get new weights
        new_w1 = w1-(pd_w1*l_rate)
        new_w2 = w2-(pd_w2*l_rate)
        new_w3 = w3-(pd_w3*l_rate)
        new_w4 = w4-(pd_w4*l_rate)
        new_w5 = w5-(pd_w5*l_rate)
        new_w6 = w6-(pd_w6*l_rate)
        new_w7 = w7-(pd_w7*l_rate)
        new_w8 = w8-(pd_w8*l_rate)
        if ignore_bias == True:
            new_w9 = 0
            new_w10 = 0
            new_w11 = 0
            new_w12 = 0
        else:
            new_w9 = w9-(pd_w9*l_rate)
            new_w10 = w10-(pd_w10*l_rate)
            new_w11 = w11-(pd_w11*l_rate)
            new_w12 = w12-(pd_w12*l_rate)
        
        #Move values to new weights for next epoch
        w1 = new_w1
        w2 = new_w2
        w3 = new_w3
        w4 = new_w4
        w5 = new_w5
        w6 = new_w6
        w7 = new_w7
        w8 = new_w8
        w9 = new_w9
        w10 = new_w10
        w11 = new_w11
        w12 = new_w12
    
        iterations_run = iterations_run +1
        euclidean_distance = np.sqrt(((w1-w_start_1)**2) + ((w2-w_start_2)**2) + ((w3-w_start_3)**2) + ((w4-w_start_4)**2) \
            + ((w5-w_start_5)**2) + ((w6-w_start_6)**2) + ((w7-w_start_7)**2) + ((w8-w_start_8)**2) + ((w9-w_start_9)**2)  \
            +  ((w10-w_start_10)**2) + ((w11-w_start_11)**2) + ((w12-w_start_12)**2))
            
    return label, error_rate, iterations_run, euclidean_distance, start_arr
    

sigmoid_and_bias = [] 

sb_1 = trainModel(0.01, 'sigmoid', False, False, 'sigmoid w/ bias - 0.01')
sigmoid_and_bias.append(sb_1)
sb_2 = trainModel(0.001, 'sigmoid', False, False, 'sigmoid w/ bias - 0.001')
sigmoid_and_bias.append(sb_2)
sb_3 = trainModel(0.0001, 'sigmoid', False, False, 'sigmoid w/ bias - 0.0001')
sigmoid_and_bias.append(sb_3)
sb_4 = trainModel(0.00001, 'sigmoid', False, False, 'sigmoid w/ bias - 0.00001')
sigmoid_and_bias.append(sb_4)
sb_5 = trainModel(0.000001, 'sigmoid', False, False, 'sigmoid w/ bias - 0.000001')
sigmoid_and_bias.append(sb_5)
print(sigmoid_and_bias)

x = [t[1] for t in sigmoid_and_bias]
y = [t[2] for t in sigmoid_and_bias]
z = [t[3] for t in sigmoid_and_bias]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Sigmoid with Bias - Iterations')
plt.show() 
plt.savefig('Sigmoid with Bias - Iterations.png') 

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Sigmoid with Bias - Euclidean Distance')
plt.show()  
plt.savefig('Sigmoid with Bias - Euclidean Distance.png') 

relu_and_bias = []

rb_1 = trainModel(0.01, 'relu', False, False, 'relu w/ bias - 0.01')
relu_and_bias.append(rb_1)
rb_2 = trainModel(0.001, 'relu', False, False, 'relu w/ bias - 0.001')
relu_and_bias.append(rb_2)
rb_3 = trainModel(0.0001, 'relu', False, False, 'relu w/ bias - 0.0001')
relu_and_bias.append(rb_3)
rb_4 = trainModel(0.00001, 'relu', False, False, 'relu w/ bias - 0.00001')
relu_and_bias.append(rb_4)
rb_5 = trainModel(0.000001, 'relu', False, False, 'relu w/ bias - 0.000001')
relu_and_bias.append(rb_5)
print(relu_and_bias)

x = [t[1] for t in relu_and_bias]
y = [t[2] for t in relu_and_bias]
z = [t[3] for t in relu_and_bias]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Relu with Bias - Iterations')
plt.show()  
plt.savefig('Relu with Bias - Iterations.png') 

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Relu with Bias - Euclidean Distance')
plt.show() 
plt.savefig('Relu with Bias - Euclidean Distance.png') 
              
sigmoid_no_bias = [] 

snb_1 = trainModel(0.01, 'sigmoid', False, True, 'sigmoid w/o bias - 0.01')
sigmoid_no_bias.append(snb_1)
snb_2 = trainModel(0.001, 'sigmoid', False, True, 'sigmoid w/o bias - 0.001')
sigmoid_no_bias.append(snb_2)
snb_3 = trainModel(0.0001, 'sigmoid', False, True, 'sigmoid w/o bias - 0.0001')
sigmoid_no_bias.append(snb_3)
snb_4 = trainModel(0.00001, 'sigmoid', False, True, 'sigmoid w/o bias - 0.00001')
sigmoid_no_bias.append(snb_4)
snb_5 = trainModel(0.000001, 'sigmoid', False, True, 'sigmoid w/o bias - 0.000001')
sigmoid_no_bias.append(snb_5)
print(sigmoid_no_bias)

x = [t[1] for t in sigmoid_no_bias]
y = [t[2] for t in sigmoid_no_bias]
z = [t[3] for t in sigmoid_no_bias]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Sigmoid without Bias - Iterations')
plt.show()  
plt.savefig('Sigmoid without Bias - Iterations.png') 

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Sigmoid without Bias - Euclidean Distance')
plt.show() 
plt.savefig('Sigmoid without Bias - Euclidean Distance.png') 

relu_no_bias = []
rnb_1 = trainModel(0.01, 'relu', False, True, 'relu w/o bias - 0.01')
relu_no_bias.append(rnb_1)
rnb_2 = trainModel(0.001, 'relu', False, True, 'relu w/o bias - 0.001')
relu_no_bias.append(rnb_2)
rnb_3 = trainModel(0.0001, 'relu', False, True, 'relu w/o bias - 0.0001')
relu_no_bias.append(rnb_3)
rnb_4 = trainModel(0.00001, 'relu', False, True, 'relu w/o bias - 0.00001')
relu_no_bias.append(rnb_4)
rnb_5 = trainModel(0.000001, 'relu', False, True, 'relu w/o bias - 0.000001')
relu_no_bias.append(rnb_5)
print(relu_no_bias)

x = [t[1] for t in relu_no_bias]
y = [t[2] for t in relu_no_bias]
z = [t[3] for t in relu_no_bias]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Relu without Bias - Iterations')
plt.show()  
plt.savefig('Relu without Bias - Iterations.png') 

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Relu without Bias - Euclidean Distance')
plt.show() 
plt.savefig('Relu without Bias - Euclidean Distance.png') 

#Randomized sigmoid batches
random_sigmoid_w_bias_1 = []
sb_r_1_1 = trainModel(0.01, 'sigmoid', True, False, 'sigmoid - random rd1 w/ bias - 0.01')
random_sigmoid_w_bias_1.append(sb_r_1_1)
sb_r_1_2 = trainModel(0.001, 'sigmoid', True, False, 'sigmoid - random rd1 w/ bias - 0.001')
random_sigmoid_w_bias_1.append(sb_r_1_2)
sb_r_1_3 = trainModel(0.0001, 'sigmoid', True, False, 'sigmoid - random rd1 w/ bias - 0.0001')
random_sigmoid_w_bias_1.append(sb_r_1_3)
sb_r_1_4 = trainModel(0.00001, 'sigmoid', True, False, 'sigmoid - random rd1 w/ bias - 0.00001')
random_sigmoid_w_bias_1.append(sb_r_1_4)
sb_r_1_5 = trainModel(0.000001, 'sigmoid', True, False, 'sigmoid - random rd1 w/ bias - 0.000001')
random_sigmoid_w_bias_1.append(sb_r_1_5)

x = [t[1] for t in random_sigmoid_w_bias_1]
y = [t[2] for t in random_sigmoid_w_bias_1]
z = [t[3] for t in random_sigmoid_w_bias_1]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Sigmoid random 1 - Iterations')
plt.show()  
plt.savefig('Sigmoid random 1 - Iterations.png') 

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Sigmoid random 1 - Euclidean Distance')
plt.show() 
plt.savefig('Sigmoid random 1 - Euclidean Distance.png') 

random_sigmoid_w_bias_2 = []
sb_r_2_1 = trainModel(0.01, 'sigmoid', True, False, 'sigmoid - random rd2 w/ bias - 0.01')
random_sigmoid_w_bias_2.append(sb_r_2_1)
sb_r_2_2 = trainModel(0.001, 'sigmoid', True, False, 'sigmoid - random rd2 w/ bias - 0.001')
random_sigmoid_w_bias_2.append(sb_r_2_2)
sb_r_2_3 = trainModel(0.0001, 'sigmoid', True, False, 'sigmoid - random rd2 w/ bias - 0.0001')
random_sigmoid_w_bias_2.append(sb_r_2_3)
sb_r_2_4 = trainModel(0.00001, 'sigmoid', True, False, 'sigmoid - random rd2 w/ bias - 0.00001')
random_sigmoid_w_bias_2.append(sb_r_2_4)
sb_r_2_5 = trainModel(0.000001, 'sigmoid', True, False, 'sigmoid - random rd2 w/ bias - 0.000001')
random_sigmoid_w_bias_2.append(sb_r_2_5)

x = [t[1] for t in random_sigmoid_w_bias_2]
y = [t[2] for t in random_sigmoid_w_bias_2]
z = [t[3] for t in random_sigmoid_w_bias_2]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Sigmoid random 2 - Iterations')
plt.show()  
plt.savefig('Sigmoid random 2 - Iterations.png') 

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Sigmoid random 2 - Euclidean Distance')
plt.show() 
plt.savefig('Sigmoid random 2 - Euclidean Distance.png') 

random_sigmoid_w_bias_3 = []
sb_r_3_1 = trainModel(0.01, 'sigmoid', True, False, 'sigmoid - random rd3 w/ bias - 0.01')
random_sigmoid_w_bias_3.append(sb_r_3_1)
sb_r_3_2 = trainModel(0.001, 'sigmoid', True, False, 'sigmoid - random rd3 w/ bias - 0.001')
random_sigmoid_w_bias_3.append(sb_r_3_2)
sb_r_3_3 = trainModel(0.0001, 'sigmoid', True, False, 'sigmoid - random rd3 w/ bias - 0.0001')
random_sigmoid_w_bias_3.append(sb_r_3_3)
sb_r_3_4 = trainModel(0.00001, 'sigmoid', True, False, 'sigmoid - random rd3 w/ bias - 0.00001')
random_sigmoid_w_bias_3.append(sb_r_3_4)
sb_r_3_5 = trainModel(0.000001, 'sigmoid', True, False, 'sigmoid - random rd3 w/ bias - 0.000001')
random_sigmoid_w_bias_3.append(sb_r_3_5)

x = [t[1] for t in random_sigmoid_w_bias_3]
y = [t[2] for t in random_sigmoid_w_bias_3]
z = [t[3] for t in random_sigmoid_w_bias_3]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Sigmoid random 3 - Iterations')
plt.show()  
plt.savefig('Sigmoid random 3 - Iterations.png') 

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Sigmoid random 3 - Euclidean Distance')
plt.show() 
plt.savefig('Sigmoid random 3 - Euclidean Distance.png') 

random_sigmoid_w_bias_4 = []
sb_r_4_1 = trainModel(0.01, 'sigmoid', True, False, 'sigmoid - random rd4 w/ bias - 0.01')
random_sigmoid_w_bias_4.append(sb_r_4_1)
sb_r_4_2 = trainModel(0.001, 'sigmoid', True, False, 'sigmoid - random rd4 w/ bias - 0.001')
random_sigmoid_w_bias_4.append(sb_r_4_2)
sb_r_4_3 = trainModel(0.0001, 'sigmoid', True, False, 'sigmoid - random rd4 w/ bias - 0.0001')
random_sigmoid_w_bias_4.append(sb_r_4_3)
sb_r_4_4 = trainModel(0.00001, 'sigmoid', True, False, 'sigmoid - random rd4 w/ bias - 0.00001')
random_sigmoid_w_bias_4.append(sb_r_4_4)
sb_r_4_5 = trainModel(0.000001, 'sigmoid', True, False, 'sigmoid - random rd4 w/ bias - 0.000001')
random_sigmoid_w_bias_4.append(sb_r_4_5)

x = [t[1] for t in random_sigmoid_w_bias_4]
y = [t[2] for t in random_sigmoid_w_bias_4]
z = [t[3] for t in random_sigmoid_w_bias_4]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Sigmoid random 4 - Iterations')
plt.show()  
plt.savefig('Sigmoid random 4 - Iterations.png') 

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Sigmoid random 4 - Euclidean Distance')
plt.show() 
plt.savefig('Sigmoid random 4 - Euclidean Distance.png') 

random_sigmoid_w_bias_5 = []
sb_r_5_1 = trainModel(0.01, 'sigmoid', True, False, 'sigmoid - random rd5 w/ bias - 0.01')
random_sigmoid_w_bias_5.append(sb_r_5_1)
sb_r_5_2 = trainModel(0.001, 'sigmoid', True, False, 'sigmoid - random rd5 w/ bias - 0.001')
random_sigmoid_w_bias_5.append(sb_r_5_2)
sb_r_5_3 = trainModel(0.0001, 'sigmoid', True, False, 'sigmoid - random rd5 w/ bias - 0.0001')
random_sigmoid_w_bias_5.append(sb_r_5_3)
sb_r_5_4 = trainModel(0.00001, 'sigmoid', True, False, 'sigmoid - random rd5 w/ bias - 0.00001')
random_sigmoid_w_bias_5.append(sb_r_5_4)
sb_r_5_5 = trainModel(0.000001, 'sigmoid', True, False, 'sigmoid - random rd5 w/ bias - 0.000001')
random_sigmoid_w_bias_5.append(sb_r_5_5)

x = [t[1] for t in random_sigmoid_w_bias_5]
y = [t[2] for t in random_sigmoid_w_bias_5]
z = [t[3] for t in random_sigmoid_w_bias_5]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Sigmoid random 5 - Iterations')
plt.show()  
plt.savefig('Sigmoid random 5 - Iterations.png') 

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Sigmoid random 5 - Euclidean Distance')
plt.show() 
plt.savefig('Sigmoid random 5 - Euclidean Distance.png') 

random_sigmoid_w_bias_6 = []
sb_r_6_1 = trainModel(0.01, 'sigmoid', True, False, 'sigmoid - random rd6 w/ bias - 0.01')
random_sigmoid_w_bias_6.append(sb_r_6_1)
sb_r_6_2 = trainModel(0.001, 'sigmoid', True, False, 'sigmoid - random rd6 w/ bias - 0.001')
random_sigmoid_w_bias_6.append(sb_r_6_2)
sb_r_6_3 = trainModel(0.0001, 'sigmoid', True, False, 'sigmoid - random rd6 w/ bias - 0.0001')
random_sigmoid_w_bias_6.append(sb_r_6_3)
sb_r_6_4 = trainModel(0.00001, 'sigmoid', True, False, 'sigmoid - random rd6 w/ bias - 0.00001')
random_sigmoid_w_bias_6.append(sb_r_6_4)
sb_r_6_5 = trainModel(0.000001, 'sigmoid', True, False, 'sigmoid - random rd6 w/ bias - 0.000001')
random_sigmoid_w_bias_6.append(sb_r_6_5)

x = [t[1] for t in random_sigmoid_w_bias_6]
y = [t[2] for t in random_sigmoid_w_bias_6]
z = [t[3] for t in random_sigmoid_w_bias_6]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Sigmoid random 6 - Iterations')
plt.show()  
plt.savefig('Sigmoid random 6 - Iterations.png') 

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Sigmoid random 6 - Euclidean Distance')
plt.show() 
plt.savefig('Sigmoid random 6 - Euclidean Distance.png') 

#Randomized relu batches
random_relu_w_bias_1 = []
rb_r_1_1 = trainModel(0.01, 'relu', True, False, 'sigmoid - random rd1 w/ bias - 0.01')
random_relu_w_bias_1.append(rb_r_1_1)
rb_r_1_2 = trainModel(0.001, 'relu', True, False, 'sigmoid - random rd1 w/ bias - 0.001')
random_relu_w_bias_1.append(rb_r_1_2)
rb_r_1_3 = trainModel(0.0001, 'relu', True, False, 'sigmoid - random rd1 w/ bias - 0.0001')
random_relu_w_bias_1.append(rb_r_1_3)
rb_r_1_4 = trainModel(0.00001, 'relu', True, False, 'sigmoid - random rd1 w/ bias - 0.00001')
random_relu_w_bias_1.append(rb_r_1_4)
rb_r_1_5 = trainModel(0.000001, 'relu', True, False, 'sigmoid - random rd1 w/ bias - 0.000001')
random_relu_w_bias_1.append(rb_r_1_5)

x = [t[1] for t in random_relu_w_bias_1]
y = [t[2] for t in random_relu_w_bias_1]
z = [t[3] for t in random_relu_w_bias_1]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Relu random 1 - Iterations')
plt.show()  
plt.savefig('Relu random 1 - Iterations.png') 

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Relu random 1 - Euclidean Distance')
plt.show() 
plt.savefig('Relu random 1 - Euclidean Distance.png') 

random_relu_w_bias_2 = []
rb_r_2_1 = trainModel(0.01, 'relu', True, False, 'sigmoid - random rd2 w/ bias - 0.01')
random_relu_w_bias_2.append(rb_r_2_1)
rb_r_2_2 = trainModel(0.001, 'relu', True, False, 'sigmoid - random rd2 w/ bias - 0.001')
random_relu_w_bias_2.append(rb_r_2_2)
rb_r_2_3 = trainModel(0.0001, 'relu', True, False, 'sigmoid - random rd2 w/ bias - 0.0001')
random_relu_w_bias_2.append(rb_r_2_3)
rb_r_2_4 = trainModel(0.00001, 'relu', True, False, 'sigmoid - random rd2 w/ bias - 0.00001')
random_relu_w_bias_2.append(rb_r_2_4)
rb_r_2_5 = trainModel(0.000001, 'relu', True, False, 'sigmoid - random rd2 w/ bias - 0.000001')
random_relu_w_bias_2.append(rb_r_2_5)

x = [t[1] for t in random_relu_w_bias_2]
y = [t[2] for t in random_relu_w_bias_2]
z = [t[3] for t in random_relu_w_bias_2]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Relu random 2 - Iterations')
plt.show()  
plt.savefig('Relu random 2 - Iterations.png') 

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Relu random 2 - Euclidean Distance')
plt.show() 
plt.savefig('Relu random 2 - Euclidean Distance.png') 


random_relu_w_bias_3 = []
rb_r_3_1 = trainModel(0.01, 'relu', True, False, 'sigmoid - random rd3 w/ bias - 0.01')
random_relu_w_bias_3.append(rb_r_3_1)
rb_r_3_2 = trainModel(0.001, 'relu', True, False, 'sigmoid - random rd3 w/ bias - 0.001')
random_relu_w_bias_3.append(rb_r_3_2)
rb_r_3_3 = trainModel(0.0001, 'relu', True, False, 'sigmoid - random rd3 w/ bias - 0.0001')
random_relu_w_bias_3.append(rb_r_3_3)
rb_r_3_4 = trainModel(0.00001, 'relu', True, False, 'sigmoid - random rd3 w/ bias - 0.00001')
random_relu_w_bias_3.append(rb_r_3_4)
rb_r_3_5 = trainModel(0.000001, 'relu', True, False, 'sigmoid - random rd3 w/ bias - 0.000001')
random_relu_w_bias_3.append(rb_r_3_5)

x = [t[1] for t in random_relu_w_bias_3]
y = [t[2] for t in random_relu_w_bias_3]
z = [t[3] for t in random_relu_w_bias_3]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Relu random 3 - Iterations')
plt.show()  
plt.savefig('Relu random 3 - Iterations.png') 

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Relu random 3 - Euclidean Distance')
plt.show() 
plt.savefig('Relu random 3 - Euclidean Distance.png')

random_relu_w_bias_4 = []
rb_r_4_1 = trainModel(0.01, 'relu', True, False, 'sigmoid - random rd4 w/ bias - 0.01')
random_relu_w_bias_4.append(rb_r_4_1)
rb_r_4_2 = trainModel(0.001, 'relu', True, False, 'sigmoid - random rd4 w/ bias - 0.001')
random_relu_w_bias_4.append(rb_r_4_2)
rb_r_4_3 = trainModel(0.0001, 'relu', True, False, 'sigmoid - random rd4 w/ bias - 0.0001')
random_relu_w_bias_4.append(rb_r_4_3)
rb_r_4_4 = trainModel(0.00001, 'relu', True, False, 'sigmoid - random rd4 w/ bias - 0.00001')
random_relu_w_bias_4.append(rb_r_4_4)
rb_r_4_5 = trainModel(0.000001, 'relu', True, False, 'sigmoid - random rd4 w/ bias - 0.000001')
random_relu_w_bias_4.append(rb_r_4_5)

x = [t[1] for t in random_relu_w_bias_4]
y = [t[2] for t in random_relu_w_bias_4]
z = [t[3] for t in random_relu_w_bias_4]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Relu random 4 - Iterations')
plt.show()  
plt.savefig('Relu random 4 - Iterations.png') 

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Relu random 4 - Euclidean Distance')
plt.show() 
plt.savefig('Relu random 4 - Euclidean Distance.png')

random_relu_w_bias_5 = []
rb_r_5_1 = trainModel(0.01, 'relu', True, False, 'sigmoid - random rd5 w/ bias - 0.01')
random_relu_w_bias_5.append(rb_r_5_1)
rb_r_5_2 = trainModel(0.001, 'relu', True, False, 'sigmoid - random rd5 w/ bias - 0.001')
random_relu_w_bias_5.append(rb_r_5_2)
rb_r_5_3 = trainModel(0.0001, 'relu', True, False, 'sigmoid - random rd5 w/ bias - 0.0001')
random_relu_w_bias_5.append(rb_r_5_3)
rb_r_5_4 = trainModel(0.00001, 'relu', True, False, 'sigmoid - random rd5 w/ bias - 0.00001')
random_relu_w_bias_5.append(rb_r_5_4)
rb_r_5_5 = trainModel(0.000001, 'relu', True, False, 'sigmoid - random rd5 w/ bias - 0.000001')
random_relu_w_bias_5.append(rb_r_5_5)

x = [t[1] for t in random_relu_w_bias_5]
y = [t[2] for t in random_relu_w_bias_5]
z = [t[3] for t in random_relu_w_bias_5]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Relu random 5 - Iterations')
plt.show() 
plt.savefig('Relu random 5 - Iterations.png')  

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Relu random 5 - Euclidean Distance')
plt.show() 
plt.savefig('Relu random 5 - Euclidean Distance.png')

random_relu_w_bias_6 = []
rb_r_6_1 = trainModel(0.01, 'relu', True, False, 'sigmoid - random rd6 w/ bias - 0.01')
random_relu_w_bias_6.append(rb_r_6_1)
rb_r_6_2 = trainModel(0.001, 'relu', True, False, 'sigmoid - random rd6 w/ bias - 0.001')
random_relu_w_bias_6.append(rb_r_6_2)
rb_r_6_3 = trainModel(0.0001, 'relu', True, False, 'sigmoid - random rd6 w/ bias - 0.0001')
random_relu_w_bias_6.append(rb_r_6_3)
rb_r_6_4 = trainModel(0.00001, 'relu', True, False, 'sigmoid - random rd6 w/ bias - 0.00001')
random_relu_w_bias_6.append(rb_r_6_4)
rb_r_6_5 = trainModel(0.000001, 'relu', True, False, 'sigmoid - random rd6 w/ bias - 0.000001')
random_relu_w_bias_6.append(rb_r_6_5)

x = [t[1] for t in random_relu_w_bias_6]
y = [t[2] for t in random_relu_w_bias_6]
z = [t[3] for t in random_relu_w_bias_6]
max_x = max(x)
max_y = max(y)
max_z = max(z)


plt.scatter(x, y)
plt.xlabel('Error Rate')
plt.ylabel('Iterations')
plt.axis([0, max_x, 0, max_y])
plt.title('Relu random 6 - Iterations')
plt.show() 
plt.savefig('Relu random 6 - Iterations.png')   

plt.scatter(x, z)
plt.xlabel('Error Rate')
plt.ylabel('Euclidean Distance')
plt.axis([0, max_x, 0, max_z])
plt.title('Relu random 6 - Euclidean Distance')
plt.show() 
plt.savefig('Relu random 6 - Euclidean Distance.png')

plt.style.use('_mpl-gallery')

