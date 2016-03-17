# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
pickle_file = '/ashwanir/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(pred, actuals):
    #print(str(np.argmax(pred[222,:])) + " " + str(np.argmax(actuals[222,:])))
    
    #print (np.sum(np.argmax(pred, 1) == np.argmax(actuals, 1)))
    return 100.00 * (np.sum(np.argmax(pred, 1) == np.argmax(actuals, 1)) / (1.0 * pred.shape[0]))

gr = tf.Graph()
batch_size = 1028
with gr.as_default():
    tf_train_data = tf.placeholder(tf.float32, shape=[None, 28 * 28])
    tf_train_wt = tf.Variable(tf.truncated_normal([28 * 28 , 1024]))
    tf_train_bias = tf.Variable(tf.zeros([1024]))
    
    tf_hidden_out = tf.nn.relu(tf.matmul(tf_train_data, tf_train_wt) + tf_train_bias)
    tf_hidden_wt = tf.Variable(tf.truncated_normal([1024, 10]))
    tf_hidden_bias = tf.Variable(tf.zeros([10]))
    
    logits = tf.matmul(tf_hidden_out, tf_hidden_wt) + tf_hidden_bias
  
    tf_train_labels = tf.placeholder(tf.float32, shape=[None, 10])
  
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    optimize = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    train_pred = tf.nn.softmax(logits)
    
    valid_dataset_c = tf.constant(valid_dataset)
    valid_dataset_hidden = tf.nn.relu(tf.matmul(valid_dataset_c, tf_train_wt) + tf_train_bias)
    valid_dataset_pred = tf.nn.softmax(tf.matmul(valid_dataset_hidden, tf_hidden_wt) + tf_hidden_bias)
    
    test_dataset_c = tf.constant(test_dataset)
    test_dataset_hidden = tf.nn.relu(tf.matmul(test_dataset_c, tf_train_wt) + tf_train_bias)
    test_dataset_pred = tf.nn.softmax(tf.matmul(test_dataset_hidden, tf_hidden_wt) + tf_hidden_bias)
    
print('Graph')

with tf.Session(graph=gr) as session:
    tf.initialize_all_variables().run()
    offset = 0
    total_rows = train_dataset.shape[0]
    while total_rows - offset > 0:
        train_batch = train_dataset[offset: offset + batch_size, :]
        train_label_batch = train_labels[offset: offset + batch_size, :]
        feed_dict = {tf_train_data : train_batch , tf_train_labels : train_label_batch}
        _, curr_loss, pred = session.run([optimize, loss, train_pred], feed_dict=feed_dict)
        print("Training data loss %.2f and accuracy %.2f"%(curr_loss,  accuracy(pred, train_label_batch)))
        print("Validation data accuracy %.2f"%(accuracy(valid_dataset_pred.eval(), valid_labels)))
       
        offset += batch_size
    print("Test data error %.2f"%(accuracy(test_dataset_pred.eval(), test_labels)))    
