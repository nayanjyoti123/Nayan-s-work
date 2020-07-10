
# coding: utf-8

# # TF Core Tutorial

# In[1]:

import tensorflow as tf


# In[2]:

tf.__version__


# In[3]:

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)


# In[4]:

sess = tf.Session()
print(sess.run([node1, node2]))


# In[5]:

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))


# A **placeholder** is a promise to provide a value later.

# In[6]:

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b


# In[7]:

print(sess.run(adder_node, {a:3, b:4.5}))
print(sess.run(adder_node, {a:[1,3], b:[2, 4]}))


# In[8]:

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b:4.5}))


# **Variables** allow us to add trainable parameters to a graph.

# In[9]:

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b


# In[10]:

init = tf.global_variables_initializer()
sess.run(init)


# In[11]:

print(sess.run(linear_model, {x:[1,2,3,4]}))


# A **loss function** measures how far apart the current model is from the provided data. We'll use a standard loss model for linear regression, which sums the squares of the deltas between the current model and the provided data. linear_model - y creates a vector where each element is the corresponding example's error delta. We call tf.square to square that error. Then, we sum all the squared errors to create a single scalar that abstracts the error of all examples using tf.reduce_sum:

# In[12]:

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))


# We could improve this manually by reassigning the values of W and b to the perfect values of -1 and 1. A variable is initialized to the value provided to tf.Variable but can be changed using operations like **tf.assign**. For example, W=-1 and b=1 are the optimal parameters for our model. We can change W and b accordingly:

# In[13]:

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))


# # tf.train API

# In[14]:

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


# In[15]:

sess.run(init) # reset values to incorrect defaults.
print(sess.run([W,b]))


# In[16]:

for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})


# In[17]:

print(sess.run([W, b]))


# # tf.contrib.learn

# In[18]:

import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np


# In[19]:

# Declare list of features. We only have one real-valued feature. 
# There are many other types of columns that are more complicated and useful.


# In[20]:

features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
print(features)


# In[21]:

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)


# In[22]:

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train,
                                              batch_size=4,
                                              num_epochs=1000)

eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
    {"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)


# In[23]:

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)


# In[24]:

# Here we evaluate how well our model did.
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)


# ### custom model

# In[29]:

import numpy as np
import tensorflow as tf


# In[30]:

# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)


# In[31]:

estimator = tf.contrib.learn.Estimator(model_fn=model)
## define our data sets
#x_train = np.array([1., 2., 3., 4.])
#y_train = np.array([0., -1., -2., -3.])
#x_eval = np.array([2., 5., 8., 1.])
#y_eval = np.array([-1.01, -4.1, -7, 0.])
#input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, 4, num_epochs=1000)
#eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)

# In[32]:

# train
estimator.fit(input_fn=input_fn, steps=1000)
# Here we evaluate how well our model did. 
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)


# In[ ]:



