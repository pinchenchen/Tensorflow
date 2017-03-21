## Tensorflow
* 官網：[Tensorflow](https://www.tensorflow.org/)
* 開源機器學習工具，神經網路的Python外部結構包
* 繪製計算數據流程圖
### 1 神經網路
* 由輸入層+隱藏層+輸出層組成
* 隱藏層輸出的output = input*weight + baise
* 輸出層 Y' 經由 activation function 激活
![](https://github.com/pinchenchen/Tensorflow/blob/master/NN.png)
* 計算 Y' 與真實數據 Y 的差距(損失函數)
* 訓練目標：訓練 Weight 與 Baise 使得 Y' 與 Y 的差距最小 
![](https://github.com/pinchenchen/Tensorflow/blob/master/speedup3.png)

### 2 基本概念
* 用`graph`描述計算的過程
* 在 `session` 中執行
* `tensor`表示數據(組)
* `Variable`變量維護執行過程中的狀態
* `feed` & `fetch`可以在任意步驟中賦值或添加數據

#### 2.1 Tensorflow基礎架構
* Computation Graph
```
# placehoder(類型, 形狀, 名子=None), 為 feed 建立佔位符, 與 feed_dict={} 一起使用
x_ = tf.placehoser( tf.float32, [None,2] )
y_ = tf.placehoser( tf.float32, [None,1] )

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

# initalizer 初始化
init = tf.initialize_all_variables()                  # 初始化所有變量
```
* Session
```
# 使用with，自動關閉
with tf.Session() as sess:
  sess.run(init)                                        # 在 session 裡使用 sess.run(init) 激活變量
  for i in xrange(500):
    sess.run(train,feed_dict={ x_:x_data, y_:y_data })  # feed_dict 搭配 placehoder 匯入input_data
    if i%50 == 0:                                       # 每50次輸出一次結果
        print i,sess.run(W),sess.run(b)                 # 輸出結果
```
### 3  Tensorboard 視覺化工具
![](https://github.com/pinchenchen/Tensorflow/blob/master/video14.png)
- 命名方式
```
...
layer_name = 'layer%s'%n_layer                                                  # 每層都命名
   with tf.name_scope(layer_name):                                              # with tf.name_scope('可打開的框框名稱')
       with tf.name_scope('Weights'):
           Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W') # 函數(..., name='節點名稱')
           tf.histogram_summary(layer_name+'/weights', Weights)                 # tf.histogram_summary()建立直方圖
...
       
merged = tf.merge_all_summaries()                                               # merge所有的summary
...

writer = tf.train.SummaryWriter("logs/", sess.graph)                            #寫入文件中儲存
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

## MNIST 
## 常用函式庫
