# MindSpore前馈神经网络运用

## 实验经典介绍

本实验主要介绍使用MindSpore开发前馈神经网络，并使用Fashion-MNIST数据集训练和测试模型。

## 实验目的

- 掌握如何使用MindSpore进行简单前馈神经网络的开发。
- 了解如何使用MindSpore进行简单图片分类任务的训练。
- 了解如何使用MindSpore进行简单图片分类任务的测试和预测。

## 预备知识

- 熟练使用Python。
- 具备一定的深度学习理论知识，如感知机、前馈神经网络、损失函数、优化器，训练策略等。
- 了解华为云的基本使用方法，包括[OBS（对象存储）](https://www.huaweicloud.com/product/obs.html)、[ModelArts（AI开发平台）](https://www.huaweicloud.com/product/modelarts.html)、[训练作业](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0238.html)等功能。华为云官网：https://www.huaweicloud.com
- 了解并熟悉MindSpore AI计算框架，MindSpore官网：https://www.mindspore.cn/

## 实验环境

- MindSpore 1.0.0（MindSpore版本会定期更新，本指导也会定期刷新，与版本配套）；
- 华为云ModelArts（控制台左上角选择“华北-北京四”）：ModelArts是华为云提供的面向开发者的一站式AI开发平台，集成了昇腾AI处理器资源池，用户可以在该平台下体验MindSpore。

## 实验准备

已经对ModelArts云环境很熟悉的玩家可以直接跳到实验步骤。

### 数据集准备

Fashion-MNIST是一个替代MNIST手写数字集的图像数据集。 它是由Zalando（一家德国的时尚科技公司）旗下的研究部门提供。其涵盖了来自10种类别的共7万个不同商品的正面图片。Fashion-MNIST的大小、格式和训练集/测试集划分与原始的MNIST完全一致。60000/10000的训练测试数据划分，28x28x1的灰度图片。

这里介绍一下经典的MNIST（手写字母）数据集。经典的MNIST数据集包含了大量的手写数字。十几年来，来自机器学习、机器视觉、人工智能、深度学习领域的研究员们把这个数据集作为衡量算法的基准之一。实际上，MNIST数据集已经成为算法作者的必测的数据集之一，但是MNIST数据集太简单了。很多深度学习算法在测试集上的准确率已经达到99.6%。

- 从[Fashion-MNIST GitHub仓库](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion)下载如下4个文件到本地并解压：

```
train-images-idx3-ubyte     training set images（47,042,560 bytes）   
train-labels-idx1-ubyte     training set labels（61,440 bytes）  
t10k-images-idx3-ubyte      test set images (7,843,840 bytes)  
t10k-labels-idx1-ubyte      test set labels (12,288 bytes) 
```



### 脚本准备

从[课程gitee仓库](https://gitee.com/mindspore/course)上下载本实验相关脚本。将脚本和数据集组织为如下形式：

```
feedforward  
├── Fashion-MNIST  
│   ├── test  
│   │   ├── t10k-images-idx3-ubyte  
│   │   └── t10k-labels-idx1-ubyte  
│   └── train  
│       ├── train-images-idx3-ubyte  
│       └── train-labels-idx1-ubyte   
└── main.py  
```

### 创建OBS桶

本实验需要使用华为云OBS存储脚本和数据集，可以参考[快速通过OBS控制台上传下载文件](https://support.huaweicloud.com/qs-obs/obs_qs_0001.html)了解使用OBS创建桶、上传文件、下载文件的使用方法（下文给出了操作步骤）。

> **提示：** 华为云新用户使用OBS时通常需要创建和配置“访问密钥”，可以在使用OBS时根据提示完成创建和配置。也可以参考[获取访问密钥并完成ModelArts全局配置](https://support.huaweicloud.com/prepare-modelarts/modelarts_08_0002.html)获取并配置访问密钥。

打开[OBS控制台](https://storage.huaweicloud.com/obs/?region=cn-north-4&locale=zh-cn#/obs/manager/buckets)，点击右上角的“创建桶”按钮进入桶配置页面，创建OBS桶的参考配置如下：

- 区域：华北-北京四
- 数据冗余存储策略：单AZ存储
- 桶名称：全局唯一的字符串
- 存储类别：标准存储
- 桶策略：公共读
- 归档数据直读：关闭
- 企业项目、标签等配置：免

### 上传文件

点击新建的OBS桶名，再打开“对象”标签页，通过“上传对象”、“新建文件夹”等功能，将脚本和数据集上传到OBS桶中。上传文件后，查看页面底部的“任务管理”状态栏（正在运行、已完成、失败），确保文件均上传完成。若失败请：

- 参考[上传对象大小限制/切换上传方式](https://support.huaweicloud.com/qs-obs/obs_qs_0008.html)，
- 参考[上传对象失败常见原因](https://support.huaweicloud.com/obs_faq/obs_faq_0134.html)。
- 若无法解决请[新建工单](https://console.huaweicloud.com/ticket/?region=cn-north-4&locale=zh-cn#/ticketindex/createIndex)，产品类为“对象存储服务”，问题类型为“桶和对象相关”，会有技术人员协助解决。

## 实验步骤

推荐使用ModelArts训练作业进行实验，适合大规模并发使用。若使用ModelArts Notebook，请参考[LeNet5](../lenet5)及[Checkpoint](../checkpoint)实验案例，了解Notebook的使用方法和注意事项。

### 代码梳理

#### 导入MindSpore模块和辅助模块

用到的框架主要包括：

- mindspore，用于神经网络的搭建 
- numpy，用于处理一些数据 
- matplotlib，用于画图、图像展示
- struct，用于处理二进制文件

```python
import os
import struct
import sys
from easydict import EasyDict as edict

import matplotlib.pyplot as plt
import numpy as np

import mindspore
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import context
from mindspore.nn.metrics import Accuracy
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore import Tensor

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
```

#### 变量定义

```python
cfg = edict({
    'train_size': 60000,  # 训练集大小
    'test_size': 10000,  # 测试集大小
    'channel': 1,  # 图片通道数
    'image_height': 28,  # 图片高度
    'image_width': 28,  # 图片宽度
    'batch_size': 60,
    'num_classes': 10,  # 分类类别
    'lr': 0.001,  # 学习率
    'epoch_size': 20,  # 训练次数
    'data_dir_train': os.path.join('Fashion-MNIST', 'train'),
    'data_dir_test': os.path.join('Fashion-MNIST', 'test'),
    'save_checkpoint_steps': 1,  # 多少步保存一次模型
    'keep_checkpoint_max': 3,  # 最多保存多少个模型
    'output_directory': './model_fashion',  # 保存模型路径
    'output_prefix': "checkpoint_fashion_forward"  # 保存模型文件名字
})
```

#### 读取并处理数据

读取数据

```python
def read_image(file_name):
    '''
    :param file_name: 文件路径
    :return:  训练或者测试数据
    如下是训练的图片的二进制格式
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    '''
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中
    head = struct.unpack_from('>IIII', file_content, 0)  # 取前4个整数，返回一个元组
    offset = struct.calcsize('>IIII')
    imgNum = head[1]  # 图片数
    width = head[2]  # 宽度
    height = head[3]  # 高度
    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'
    imgs = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    imgs_array = np.array(imgs).reshape((imgNum, width * height))  # 最后将读取的数据reshape成 【图片数，图片像素】二维数组
    return imgs_array


def read_label(file_name):
    '''
    :param file_name:
    :return:
    标签的格式如下：
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    '''
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中
    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')
    labelNum = head[1]  # label数
    bitsString = '>' + str(labelNum) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)


def get_data():
    # 文件获取
    train_image = os.path.join(cfg.data_dir_train, 'train-images-idx3-ubyte')
    test_image = os.path.join(cfg.data_dir_test, "t10k-images-idx3-ubyte")
    train_label = os.path.join(cfg.data_dir_train, "train-labels-idx1-ubyte")
    test_label = os.path.join(cfg.data_dir_test, "t10k-labels-idx1-ubyte")
    # 读取数据
    train_x = read_image(train_image)
    test_x = read_image(test_image)
    train_y = read_label(train_label)
    test_y = read_label(test_label)
    return train_x, train_y, test_x, test_y
```

数据预处理和处理结果图片展示

```python
train_x, train_y, test_x, test_y = get_data()
train_x = train_x.reshape(-1, 1, cfg.image_height, cfg.image_width)
test_x = test_x.reshape(-1, 1, cfg.image_height, cfg.image_width)
train_x = train_x / 255.0
test_x = test_x / 255.0
train_x = train_x.astype('Float32')
test_x = test_x.astype('Float32')
train_y = train_y.astype('int32')
test_y = test_y.astype('int32')
print('训练数据集样本数：', train_x.shape[0])
print('测试数据集样本数：', test_y.shape[0])
print('通道数/图像长/宽：', train_x.shape[1:])
print('一张图像的标签样式：', train_y[0])  # 一共10类，用0-9的数字表达类别。

plt.figure()
plt.imshow(train_x[0,0,...])
plt.colorbar()
plt.grid(False)
plt.show()
```

    训练数据集数量： 60000
    测试数据集数量： 10000
    通道数/图像长/宽： (1, 28, 28)
    一张图像的标签样式： 9

![png](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202203161738698.png)

使用MindSpore  GeneratorDataset接口将numpy.ndarray类型的数据转换为Dataset

```python
# 转换数据类型为Dataset
XY_train = list(zip(train_x, train_y))
ds_train = ds.GeneratorDataset(XY_train, ['x', 'y'])
ds_train = ds_train.shuffle(buffer_size=cfg.train_size).batch(cfg.batch_size, drop_remainder=True)
XY_test = list(zip(test_x, test_y))
ds_test = ds.GeneratorDataset(XY_test, ['x', 'y'])
ds_test = ds_test.shuffle(buffer_size=cfg.test_size).batch(cfg.batch_size, drop_remainder=True)
```

#### 定义前馈神经网络

前馈神经网络是一种最简单的神经网络，各神经元分层排列（其中每一层包含若干个神经元）。每个神经元只与前一层的神经元相连，接收前一层的输出，并输出给下一层，各层间没有反馈。是目前应用最广泛、发展最迅速的人工神经网络之一。第0层叫输入层，最后一层叫输出层，其他中间层叫做隐含层（或隐藏层、隐层）。隐层可以是一层，也可以是多层，是由全连接层堆叠而成。

![](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202203161826416.png)

<div align=center>
    <img src="./images/input_1.png" width="600"/>
</div>


```python
# 定义前馈神经网络
class Forward_fashion(nn.Cell):
    def __init__(self, num_class=10):  # 一共分十类，图片通道数是1
        super(Forward_fashion, self).__init__()
        self.num_class = num_class
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(cfg.channel * cfg.image_height * cfg.image_width, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(128, self.num_class)

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

#### 训练

使用Fashion-MNIST数据集对上述定义的前馈神经网络模型进行训练。训练策略如下表所示，可以调整训练策略并查看训练效果。

| batch size | number of epochs | learning rate | input shape | optimizer |
| :--------: | :--------------: | :-----------: | :---------: | :-------: |
|     60     |        20        |     0.001     | (1，28，28) |   Adam    |

```python
# 构建网络
network = Forward_fashion(cfg.num_classes)
# 定义模型的损失函数，优化器
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
net_opt = nn.Adam(network.trainable_params(), cfg.lr)
# 训练模型
model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={"acc"})
loss_cb = LossMonitor(per_print_times=int(cfg.train_size / cfg.batch_size))
config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                             keep_checkpoint_max=cfg.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix=cfg.output_prefix, directory=cfg.output_directory, config=config_ck)
print("============== Starting Training ==============")
model.train(cfg.epoch_size, ds_train, callbacks=[ckpoint_cb, loss_cb], dataset_sink_mode=False)
```

    ============== Starting Training ==============
    epoch: 1 step: 1000, loss is 0.6812696
    epoch: 2 step: 1000, loss is 0.39710096
    epoch: 3 step: 1000, loss is 0.43427807
    epoch: 4 step: 1000, loss is 0.3170758
    epoch: 5 step: 1000, loss is 0.24550956
    epoch: 6 step: 1000, loss is 0.4204946
    epoch: 7 step: 1000, loss is 0.35653585
    epoch: 8 step: 1000, loss is 0.31376493
    epoch: 9 step: 1000, loss is 0.27455378
    epoch: 10 step: 1000, loss is 0.18871705
    epoch: 11 step: 1000, loss is 0.20512795
    epoch: 12 step: 1000, loss is 0.2589024
    epoch: 13 step: 1000, loss is 0.31454447
    epoch: 14 step: 1000, loss is 0.24145015
    epoch: 15 step: 1000, loss is 0.32082427
    epoch: 16 step: 1000, loss is 0.27023837
    epoch: 17 step: 1000, loss is 0.34484679
    epoch: 18 step: 1000, loss is 0.41191268
    epoch: 19 step: 1000, loss is 0.07990202
    epoch: 20 step: 1000, loss is 0.26586318

![image-20220316175134855](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202203161751930.png)

#### 评估测试

```python
# 使用测试集评估模型，打印总体准确率
metric = model.eval(ds_test, dataset_sink_mode=False)
print(metric)
```

![image-20220316175152550](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202203161751586.png)

#### 预测

```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#从测试集中取出一组样本，输入模型进行预测
test_ = ds_test.create_dict_iterator()._get_next()
#利用key值选出样本
test = Tensor(test_['x'], mindspore.float32)
predictions = model.predict(test)
softmax = nn.Softmax()
predictions = softmax(predictions)

predictions = predictions.asnumpy()
true_label = test_['y'].asnumpy()
true_image = test_['x'].asnumpy()

for i in range(15):
    p_np = predictions[i, :]
    pre_label = np.argmax(p_np)
    print('第' + str(i) + '个sample预测结果：', class_names[pre_label], '   真实结果：', class_names[true_label[i]])
```

![image-20220316175222832](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202203161752879.png)

#### 对预测结果可视化

```python
# -------------------定义可视化函数--------------------------------
# 输入预测结果序列，真实标签序列，以及图片序列
# 目标是根据预测值对错，让其标签显示为红色或者蓝色。对：标签为蓝色；错：标签为红色
def plot_image(predicted_label, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    # 显示对应图片
    plt.imshow(img, cmap=plt.cm.binary)
    # 显示预测结果的颜色，如果对上了是蓝色，否则为红色
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    # 显示对应标签的格式，样式
    plt.xlabel('{},({})'.format(class_names[predicted_label],
                                    class_names[true_label]), color=color)
```

```python
# 将预测的结果以柱状图形状显示蓝对红错
def plot_value_array(predicted_label, true_label,predicted_array):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    this_plot = plt.bar(range(10), predicted_array, color='#777777')
    plt.ylim([0, 1])
    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')
```

```python
# 预测15个图像与标签，并展现出来
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    pred_np_ = predictions[i, :]
    predicted_label = np.argmax(pred_np_)
    image_single = true_image[i, 0, ...]
    plot_image(predicted_label, true_label[i], image_single)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(predicted_label, true_label[i], pred_np_)
plt.show()
```

![png](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202203161752139.png)

### 

### ![img](data:image/png;base64,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7397dZ555hkvtmHDBi82bty4QHnOnDleHW3Nkdau3r17ezEAAND9cKcYAAAAABBZTIoBAAAAAJHFpBgAAAAAEFmsKcYeW79+faBcUFDg1TnyyCMT1BqgZwi7hvUPf/hDoPzAAw94dQYNGuTFRo4c6cXefffdQHnZsmVenb59+4Zql9v+Aw44wKvz5ptverGjjz7ai+3cuTNQXrlypVdH+6kFbR3zmjVrAuXLL7/cq7Nw4UIvpr1vN+N1IjKrAgCA9sedYgAAAABAZDEpBgAAAABEFpNiAAAAAEBksaYYAADAUVgoUlISv15uroiyRB8A0I0wKe6iqqqqvFh5eXmgXFRUFLeOiMjYsWO9WH19faCsJYhJTk72YnvvvbcXu+SSS+Iej0RbQMvchE0ievIqzUsvvRQo5+XleXXS0tK8WGVlpRcbNWpUoNynj/8norq62os1NjZ6sZqamkB548aNXp3hw4d7sbVr13ox91xoSbW0trr9nIjfr2nn4Z577vFis2fP9mK7du0KlLVkXOieCgtFxo0Tqa2NXzclRWTVKibGANCd8fg0AABAMyUl4SbEIrZemDvKAICui0kxAAAAACCymBQDAAAAACKLSTEAAAAAILJItNWOGhoavJibIGb16tVenauvvtqLbdq0yYu5CW60hDduchsRPQGNG+vXr59Xp1cv/zuT7OxsL+Ymxrn00ku9OgBatnPnTi+mJbr7/PPPvZiblC83N9erUxtycaS7r0GDBnl1cnJyvFhpaakXc5Nc1dXVeXWysrK8mJb0Lz09PVBesWKFV0d731riqx07dsTd7oUXXvBiWqItd/9awjTt/QAAgK6FO8UAAAAAgMhiUgwAAAAAiCwmxQAAvkrWgAAAIABJREFUAACAyGJSDAAAAACILBJttaPGxkYv5iba0up89dVXXiwtLc2LuUldtCQ12nZaAjB3W61dqampXkxLlpOfnx8on3baaV4dAC0Lm4zpkUce8WJuQistQd6WLVu82PDhw72Y28do/YJGS+bn0valJRjT2u8mANP6OY2WKMxNtOX20SJ6n7l+/Xov5p7DMH8DAABA18OdYgAAAABAZDEpBgAAAABEFpNiAAAAAEBksaa4HYVZOzZixAgvlp2d7cW0tXbuur3t27d7dWpqaryYtjbYXcPortkT0dfHabF+/fp5MQAtM8YEyu5a3pY8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)

### 适配训练作业(可跳过)

创建训练作业时，运行参数会通过脚本传参的方式输入给脚本代码，脚本必须解析传参才能在代码中使用相应参数。如data_url和train_url，分别对应数据存储路径(OBS路径)和训练输出路径(OBS路径)。脚本对传参进行解析后赋值到args变量里，在后续代码里可以使用。

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
args, unknown = parser.parse_known_args()
```

MindSpore暂时没有提供直接访问OBS数据的接口，需要通过ModelArts自带的moxing框架与OBS交互。将OBS桶中的数据拷贝至执行容器中，供MindSpore使用：

```python
import moxing
# src_url形如's3://OBS/PATH'，为OBS桶中数据集的路径，dst_url为执行容器中的路径
moxing.file.copy_parallel(src_url=args.data_url, dst_url='Fashion-MNIST/')
```

如需将训练输出（如模型Checkpoint）从执行容器拷贝至OBS，请参考：

```python
import moxing
# src_url为执行容器中的路径，dst_url形如's3://OBS/PATH'，目录若不存在则会新建
moxing.file.copy_parallel(src_url='model_fashion', dst_url=args.train_url)   
```

### 创建训练作业

可以参考[使用常用框架训练模型](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0238.html)来创建并启动训练作业。

打开[ModelArts控制台-训练管理-训练作业](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/trainingJobs)，点击“创建”按钮进入训练作业配置页面，创建训练作业的参考配置：

>- 算法来源：常用框架->Ascend-Powered-Engine->MindSpore；
>- 代码目录：选择上述新建的OBS桶中的feedforward目录；
>- 启动文件：选择上述新建的OBS桶中的feedforward目录下的`main.py`；
>- 数据来源：数据存储位置->选择上述新建的OBS桶中的feedforward目录下的Fashion-MNIST目录；
>- 训练输出位置：选择上述新建的OBS桶中的feedforward目录并在其中创建model_fashion目录；
>- 作业日志路径：同训练输出位置；
>- 规格：Ascend:1*Ascend 910；
>- 其他均为默认；

>**启动并查看训练过程：**
>
>1. 点击提交以开始训练；
>2. 在训练作业列表里可以看到刚创建的训练作业，在训练作业页面可以看到版本管理；
>3. 点击运行中的训练作业，在展开的窗口中可以查看作业配置信息，以及训练过程中的日志，日志会不断刷新，等训练作业完成后也可以下载日志到本地进行查看；
>4. 参考上述代码梳理，在日志中找到对应的打印信息，检查实验是否成功；

![image-20220316180417163](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202203161804257.png)

![image-20220316180903418](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202203161809457.png)

![image-20220316182448025](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202203161824082.png)

运行成功。

## 实验小结

本实验展示了如何使用MindSpore进行Fashion-MNIST数据集分类。

首先训练前馈神经网络，然后使用训练后的前馈神经网络模型对Fashion-MNIST测试数据进行分类，从结果上分析准确率大于80%，即前馈神经网络学习到了Fashion-MNIST数据集分类。
