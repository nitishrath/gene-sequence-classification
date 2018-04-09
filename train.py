#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from model import CNN
from sklearn.utils import shuffle
from tensorflow.contrib import learn
import matplotlib.pyplot as plt




# Model Hyperparameters

tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_string("filter_sizes", '3,4,5', "Comma-separated filter sizes (default: '3,4,5')")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 15000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


#data loading

print("Loading data...")
ftrain=open("train1.csv","r")
ftest=open("test1.csv","r")

A=ftrain.read()
A=A.strip().split("\n")
x_train=[]
y_train=[]

for i in A:
    i=i.split(",")
    x_train.append(i[:-2])
    y_train.append(i[-2:])
A=ftest.read()
A=A.strip().split("\n")
#print len(x_train[-1])
x_dev=[]
y_dev=[]

for i in A:
    i=i.split(",")
    x_dev.append(i[:-2])
    y_dev.append(i[-2:])

ftrain.close()
ftest.close()

x_train+=x_dev
y_train+=y_dev
print len(y_train)

# Randomly shuffle data

x_shuffled,y_shuffled = shuffle(x_train,y_train)
loss_aggregate=[]


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNN(
            sequence_length=14,
            num_classes=2,
            embedding_size=4,
            filter_sizes=FLAGS.filter_sizes,
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))


        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)


        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        def batch_iter(data, batch_size, num_epochs, shuffle=True):
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
            for epoch in range(num_epochs):
                # Shuffle the data at each epoch
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffled_data = data[shuffle_indices]
                else:
                    shuffled_data = data
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    yield shuffled_data[start_index:end_index]

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
            return step,loss

        batches = batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        step_epoch=(int)(2000/FLAGS.batch_size)+1
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch=np.resize(np.array(x_batch),(np.shape(x_batch)[0],14,4,1))
            y_batch=np.resize(np.array(y_batch),(np.shape(x_batch)[0],2))
            step,loss=train_step(x_batch, y_batch)
            if step %(50*step_epoch)==0:
                loss_aggregate.append( loss)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
x=range(len(loss_aggregate))
x=[i*50 for i in x]
plt.plot(x, loss_aggregate)
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Loss vs Epochs')
plt.grid(True)
plt.savefig("fig"+(str)(FLAGS.num_epochs) +".png")
plt.show()