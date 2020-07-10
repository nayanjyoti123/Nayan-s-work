
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np


# In[2]:

def incr(k):
    with tf.variable_scope("conv1"):
        tb = tf.get_variable("b", (2,2), initializer=tf.constant_initializer(3))
        x = tb
        for _ in range(k):
            #print("AG")
            x = x + 1
        
        tupdate = tf.assign(tb, x)
        
    return (tb, tupdate, x)


# In[3]:

with tf.Session() as sess:
    
    a = np.array([[2,3,4], [5,6,7]])
    
    ta = tf.convert_to_tensor(a)
    print("ta : {0}".format(ta))
    print("---------")
    
    tc = tf.get_variable("c", (3,3), initializer=tf.constant_initializer(1))
    tc2 = tc + 1
    tc3 = tf.add(tc, tf.constant(1.0))
    update = tf.assign(tc, tc3)
    print("tc : {0}".format(tc))
    print("tc2 : {0}".format(tc2))
    print("tc3 : {0}".format(tc3))
    print("update : {0}".format(update))
    
    sess.run(tf.variables_initializer([tc]))
    print(sess.run(tc))
    print(sess.run(update))

    
    tb, tu, tx = incr(10)
    print("tb : {0}".format(tb))
    print("tu : {0}".format(tu))
    print("tx : {0}".format(tx))
    print("---------")
    
    sess.run(tf.variables_initializer([tb]))
    print(sess.run(tb))
    print(sess.run(tx))
    
    print(sess.run(tu))
    print(sess.run(tx))
    sess.run(tu)
    print(sess.run(tb))
    print("---------")
    
    tf.get_variable_scope().reuse_variables()
    tb2, tu2, tx2 = incr(1000)
    print(sess.run(tb2))
    print(sess.run(tu2))
    print(sess.run(tb2))
    print(sess.run(tx2))
    print("---------")


# In[ ]:



