## dropout解决过拟合 2017.9.21
- 在训练时，减少权重的数量(随机把部分权重置为0)，可以防止过拟合
- 对应tensorflow里的函数tf.nn.dropout
    > dropout(  
     x,     
    keep_prob,      
    noise_shape=None,       
    seed=None,      
    name=None       
)

>- x: A tensor.
>- keep_prob: A scalar Tensor with the same type as x. The probability that each element is kept.


### 例子
训练、验证数据为手写十个数字的图片数据，大小8*8，建立三层神经元层，神经元个数分别为64,50,10，在没有用dropout的情况下出现了过拟合现象，在用了dropout下，keepdrop设为0.5，过拟合消失，下面是具体实现代码：

```
# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b )
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
# summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)


init = tf.global_variables_initializer()
sess.run(init)
for i in range(500):
    # here to determine the keeping probability
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
```

### 不懂之处
```
 train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
```
在记录数据时,keep_prob为什么要一直是1，不是有用0.5在训练数据吗？？

### 参考资料链接：
https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-02-dropout/
https://www.tensorflow.org/api_docs/python/tf/nn/dropout

<br>
<br>

##  CNN卷积神经网络 2017.9.22
### 卷积神经网络训练手写数字图片过程
1. input,载入图片数据。([None,784])
2. reshape,把输入的维度转换训练需要的维度。([Node,784]---->[None,28,28,1])
3. convolution2d，进行卷积，卷积核个数为32个，卷积核大小5*5，步长为1，使用填充算法(填充算法大概指从第一个开始训练，边缘没有数据则全部填充0，不会影响卷积结果)。([None,28,28,1]-->卷积,核为[5,5,1,32]-->[None,28,28,32])
4. pooling(池化),池化的目的是为了提取最有用的特征，降低维度。池化的方法一般有两种：最大池化和平均池化。这里选择最大池化，池化窗口大小为2*2，步长为2，使用填充算法。([None,28,28,32]-->池化[1,2,2,1]-->[None,14,14,32])
5. convolution2d，再次进行卷积，卷积核个数为64个，卷积核大小5*5，步长为1，使用填充算法(填充算法大概指从第一个开始训练，边缘没有数据则全部填充0，不会影响卷积结果)。([None,14,14,32]-->卷积,核为[5,5,32,64]-->[None,14,14,64])
6. pooling,池化窗口大小为2*2，步长为2，使用填充算法。([None,14,14,64]-->池化[1,2,2,1]-->[None,7,7,64])
7. reshape,平铺矩阵,向量化，开始进行全连接训练。([None,7,7,64]-->平铺)
8. 分类，softmax函数。
9. 开始重复训练模型，降低误差，请注意训练期间的过拟合发生，神经网络极易发生过拟合。

#### 代码如下
```
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
#计算模型在验证集上的精度
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

#卷积核
def weight_variable(shape):
    inital=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital)
#偏置单元
def bias_variable(shape):
    inital=tf.constant(0.1,shape=shape)
    return tf.Variable(inital)
#卷积过程
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#池化过程
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#输入数据
xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
#reshape
x_image=tf.reshape(xs,[-1,28,28,1])

#conv1
W_conv1=weight_variable([5,5,1,32])
b_bias1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_bias1)
h_pool1=max_pool(h_conv1)  #32*14*14

#conv2
W_conv2=weight_variable([5,5,32,64])
b_bias2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_bias2)
h_pool2=max_pool(h_conv2) #64*7*7

#fc1
W_fc1=weight_variable([64*7*7,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,64*7*7])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)


#fc2
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2 )


cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                            axis=1))

train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess=tf.Session()

init=tf.global_variables_initializer()
sess.run(init)

for i in range(20001):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i % 1000 ==0:
        #输出模型精度
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))
```

### 参考资料
1. [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)
2. [tf.nn.max_pool](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool)
3. [卷积神经网络简介](https://zhuanlan.zhihu.com/p/25249694)
4. [3D Visualization of a Convolutional Neural Network](http://scs.ryerson.ca/~aharley/vis/conv/)
5. [CNN 卷积神经网络](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-05-CNN3/)
6. [莫烦系列教程 - Tensorflow教程(35)](https://www.bilibili.com/video/av12556478/index_20.html#page=35)
7. [A Guide to TF Layers: Building a Convolutional Neural Network](https://www.tensorflow.org/tutorials/layers)
8. [full_code.py](https://www.tensorflow.org/tutorials/layers)


## RNN循环神经网络 2017.9.24
### 参考资料
- [RNN 循环神经网络](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-07-RNN1/)
- [理解LSTM网络](http://www.jeyzhang.com/understanding-lstm-network.html)