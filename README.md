## Tensorflow
* 官網：[Tensorflow](https://www.tensorflow.org/)
* 開源機器學習工具，神經網路的Python外部結構包
* 繪製計算數據流程圖
### 0 安裝
* Tensorflow1.0.1 CPU
* Linux with Anaconda and jupyter notebook
* ( pip uninstall tensorflow)(pip uninstall protobuf)
1. download Anaconda
2. download Tensorflow
```
    $ conda create -n tensorflow
    $ source activate tensorflow
    $ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp27-none-linux_x86_64.whl
```
3. use jupyter notebook
```
$ conda install ipython`
$ conda install jupyter`
$ ipython kernelspec install-self --user
```
I get `Installed kernelspec python2 in /root/.local/shere/Jupyter/kernels/python2`
```
$ mkdir -p ~/.ipython/kernels
$ mv ~/.local/share/jupyter/kernels/python2 ~/.ipython/kernels/tfkernels
cd ~/.ipython/kernels/tfkernel/
```
* problem

下載packages的時候，與anaconda調用的路徑不一樣，需要把packages的檔案複製到anaconda路徑裡。
anaconda路徑:
```
import sys
sys.path
```
`/root/anaconda2/envs/tensorflow/lib/python2.7/site-pacages`

### 1 神經網路
* 由輸入層+隱藏層+輸出層組成
* 隱藏層輸出的output = input\*weight + baise
* 輸出層 Y' 經由 activation function 激活
![](https://github.com/pinchenchen/Tensorflow/blob/master/NN.png)
* 計算 Y' 與真實數據 Y 的差距(損失函數)
* 訓練目標：訓練 Weight 與 Baise 使得 Y' 與 Y 的差距最小(↓不同方法的訓練過程)
![](https://github.com/pinchenchen/Tensorflow/blob/master/speedup3.png)

### 2 Tensorflow基本概念
* 用`graph`描述計算的過程
* 在 `session` 中執行
* `tensor`表示數據(組)
* `Variable`變量維護執行過程中的狀態
* `feed` & `fetch`可以在任意步驟中賦值或添加數據

#### 2.1 Tensorflow基礎架構
* Computation Graph
```
# placeholder(類型, 形狀, 名子=None), 為 feed 建立佔位符, 與 feed_dict={} 一起使用
x_ = tf.placeholder( tf.float32, [None,2] )
y_ = tf.placeholder( tf.float32, [None,1] )

# Variable 創建變量
w = tf.Variable( tf.random_uniform([2,1],-1.0,1.0) )  # Weight 範圍是 -1.0 ~ 1.0
b = tf.Variable( tf.zeros([1.1]) )                    # biases 初始值為0

# operations 定義激活函數
y = tf.matmul(x_,w)+b                                 # 這裡 activation function is None

# error function 定義損失函數
loss =  tf.reduce_mean(tf.square(y-y_data))

# trainer 訓練
optimizer = tf.train.GradientDescentOptimizer(0.5)    # 使用GradientDescent方法，學習速率是0.5
train = optimizer.minimize(loss)                      # 使用梯度下降法優化loss以找到最小值

# initalizer 初始化所有變量( 不同的 tensorflow 版本有差( 0.12版前後 )
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
```
* Session
```
# 使用with，自動關閉
with tf.Session() as sess:
  sess.run(init)                                        # 在 session 裡使用 sess.run(init) 激活變量
  for i in xrange(500):
    sess.run(train,feed_dict={ x_:x_data, y_:y_data })  # feed_dict 搭配 placehoder 匯入input_data
    if i%50 == 0:                                       # 每50次輸出一次結果
        print i,sess.run(w),sess.run(b)                 # 輸出結果
```
### 3  Tensorboard 視覺化工具
![](https://github.com/pinchenchen/Tensorflow/blob/master/video14.png)
- 命名方式
```
...
layer_name = 'layer%s'%n_layer                                                  # 每層都命名
   with tf.name_scope(layer_name):                                              # with tf.name_scope
       with tf.name_scope('Weights'):                                             ('可打開的框框名稱')
           Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W') # 函數(..., name='橢圓節點名稱')
           tf.histogram_summary(layer_name+'/weights', Weights)                 # tf.histogram_summary()
...                                                                                            建立直方圖
       
merged = tf.merge_all_summaries()                                               # merge所有的summary
...
# 寫入文件中儲存 ( 不同的 tensorflow 版本有差( 0.12版前後 )
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    writer = tf.train.SummaryWriter('logs/', sess.graph)
else: # tensorflow version >= 0.12
    writer = tf.summary.FileWriter("logs/", sess.graph)                           
...

for i in range(500):
   sess.run(train,feed_dict={x_:x_data,y_:y_data})
   if i%50==0:
       summary = sess.run(merged, feed_dict={x_: x_data, y_:y_data})            #每50步顯示一次 merged 結果
       writer.add_summary(summary, i)
```
- Launch Tensorboard
```
$ tensorboard --logdir=./logs/
```
get http://localhost:6006

## 神經網路架構
* 建構神經網路
```
from __future__ import print_function
import tensorflow as tf
import numpy as np

# placeholder 提供佔位符給輸入資料
# placeholder(符點數, [n*1]維度, 名子)                    
with tf.name_scope('inputs'):                                     # inputs框框命名為'inputs'
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')    # None:不設定
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')    # ys input節點命名為'y_input'
    
# 建構隱藏神經層 layer 與創建變量 Variable 
# add_layer( input, 輸入維度, 輸出維度, 激活函數 )
def add_layer(inputs, in_size, out_size, activation_function=None):
  with tf.name_scope('layer'):                                                  
     with tf.name_scope('weights'):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')  # weight變量節點命名為'W'
        tf.histogram_summary(layer_name+'/weights', Weights)                    # tf.histogram_summary()建立直方圖
     with tf.name_scope('biases'):                                              # biase框框命名為'biases'
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')           # biase起始值為0.1,大小為[1*out_size]
        tf.histogram_summary(layer_name+'/biases', biases)                       
     with tf.name_scope('Wx_plus_b'):                                           # 輸出的框框命名為'Wx_plus_b'
        Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
     
     if activation_function is None:                                            # 無激活函數則 outputs = Wx_plus_b
        outputs = Wx_plus_b
     else:
        outputs = activation_function(Wx_plus_b, ) 
        
     tf.histogram_summary(layer_name+'/outputs',outputs)                        # tf.histogram_summary()建立直方圖
     return outputs
     
# 建立隱藏層 (input=xs, 輸入維度=1 → 輸出維度=10, 激活函數=relu function)
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# 定義輸出層 (input=l1 隱藏層的輸出, 輸入維度=10 → 輸出維度=1, 激活函數=None)
prediction = add_layer(l1, 10, 1, activation_function=None)

# 計算 loss 損失函數 (對 ys & prediction 之間的差 取平方和，再將第二個維度相加 取平均值)
loss = tf.reduce_mean( tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]) )

# 使用 GradientDescent 梯度下降法最小化loss
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 定義 session 
with tf.Session() as sess:

    # Summary 寫入文件中儲存
    writer = tf.train.SummaryWriter('logs/', sess.graph) 

    # 初始化所有變量
    sess.run(tf.initialize_all_variables())

    # 執行 session (feed_dict 搭配 placehoder 匯入input_data)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i%100==0:
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))

# 更新 Summary
writer = tf.summary.FileWriter("logs/", sess.graph)
```
```
$ tensorboard --logdir='logs/'
```
## 建立一個 y = x^2 + 2x + 1 的函數模型
* problem
1. 解決: no module named matplotlib

`conda install matplotlib`

`sudo yum install python-matplotlib`

將 `anaconda2/pkgs/matplotlib-2.0.0-np111py27_0`裡面所有的檔案複製到 `/root/anaconda2/envs/tensorflow/lib/python2.7/site-pacages`

2. 為解決: jupyter notebook內跑出的圖只有x,y資料集，沒有預測線

但在直接用python是有預測線的

* 完整程式碼
```
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Make up some real data
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + 2*x_data + 1 + noise

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
    
# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
```
## MNIST 
* MNIST
  + MNIST資料集 = 55,000 筆訓練集 + 10,000 筆測試集 + 5,000 筆驗證數據集 
  + 每個圖像大小是 28 x 28 = 784 ，分別對應到數字 0 ~ 9 ，所以總共有 10 個 Laybel
![](https://github.com/pinchenchen/Tensorflow/blob/master/MNIST_NUMBER.png)
* import
```
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
```
* Download MNIST
```
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```
* Defined Neural Network Function
```
def NeuralNetwork():
    # 圖像大小是 28 x 28 = 784，總共有 10 個 Laybel
    # [None,784]：將每一張圖都轉為 None(任意)*784 的維度
    xs = tf.placeholder(tf.float32,[None,784])
    ys = tf.placeholder(tf.float32,[None,10])
    
    # activation_function=tf.nn.softmax：透過 Softmax 函數將分類器輸出的分數(Evidence)轉換為機率(Probability)
    prediction = add_layer(xs, 784, 10, n_layer=1,activation_function=tf.nn.softmax)
    
    # 計算loss
    loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
    # 使用梯度下降法優化Loss，學習速率是0.5
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    # Run the Session
    sess = tf.Session()
    # 初始化所有變量
    sess.run(tf.initialize_all_variables())
    # 跑2000次，每200次print準確率
    # batch(100)：每次隨機抓取100筆去做訓練
    for i in range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
        if i%200==0:
            print(compute_accuracy(xs,ys,mnist.test.images, mnist.test.labels,sess,prediction))
```
* Defined Layer
```
def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    # 每一層都為他命名
    layer_name = 'layer%s'%n_layer
    
    # 每層的內容都有 Weight, biases, Ws_plus_b
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):      
            # Weight 是一個[in_size*out_size]形狀的隨機變數
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            # 建立直方圖
            tf.histogram_summary(layer_name+'/weights', Weights)
        with tf.name_scope('biases'):
            # biases 形狀為[1*out_size]，內容是0.1
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
            tf.histogram_summary(layer_name+'/biases',biases)
        with tf.name_scope('Ws_plus_b'):
            # Ws_plus_b = input值* Weights + biases
            Ws_plus_b = tf.matmul(inputs,Weights) + biases
        
        # 激活函數，預設為None
        if activation_function is None:
            outputs = Ws_plus_b
        else:
            outputs = activation_function(Ws_plus_b)
        tf.histogram_summary(layer_name+'/outputs',outputs)
        return outputs
```
* Compute Accuracy
```
def compute_accuracy(xs,ys,v_xs,v_ys,sess,prediction):
    # feed_dict={xs:v_xs}搭配placeholder將資料匯入
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    # tf.argmax()：得到每一筆資料相應的label值，並用 tf.equal() 檢查實際與預測是否相等(結果是布林值)
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    # tf.cast()：將布林值轉為0/1。並取平均數。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result
```
* Run
```
NeuralNetwork()
```
## 常用函式庫

