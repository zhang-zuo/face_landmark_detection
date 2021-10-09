# PFLD - 人脸5点关键点检测

## 资料 ##
* 更多信息：[CSDN博客](https://blog.csdn.net/zz_0000000/article/details/120673679?spm=1001.2014.3001.5501)

* 数据集：[百度云盘](https://pan.baidu.com/s/12ZjbHJppklQunrXqtfv5Mg)  密码：jc6w

## 网络结构 ##

#### backbone主干网络 ####

PFLD框架的基础网络是基于MoblieNet V2进行修改的，在主干网络中使用了Inverted Residual Block基础模块和深度可分离卷积：

* Inverted Residual Block基础模块是由1x1,3x3,1x1三个卷积构成的残差网络。

* 深度可分离卷积是将输入的通道分组进行卷积（以下简称DW）。

这样可以保持网络性能的同事，减少网络的参数、运算量和计算时间。PFLD基础网络如下：

|                      Imput                       |          Operator           |       channel        |     number      |     stride      |
| :----------------------------------------------: | :-------------------------: | :------------------: | :-------------: | :-------------: |
|                    112x112x3                     |           Conv3x3           |          64          |        1        |        2        |
|                     56x56x64                     |         DW Conv3x3          |          64          |        1        |        1        |
|                     56x56x64                     |   Inverted Residual Block   |          64          |        5        |        2        |
|                     28x28x64                     |   Inverted Residual Block   |         128          |        1        |        2        |
|                    14x14x128                     |   Inverted Residual Block   |         128          |        6        |        1        |
|                    14x14x128                     |   Inverted Residual Block   |          16          |        1        |        1        |
| (S1) 14x14x16<br />(S2) 7x7x32<br />(S3) 1x1x128 | Conv3x3<br />Conv7x7<br />- | 32<br />128<br />128 | 1<br />1<br />1 | 2<br />1<br />- |
|                     S1,S2,S3                     |       Full Connection       |         136          |        1        |        -        |

#### loss损失函数 ####

损失函数没有使用PFLD论文中的结合人脸姿态角信息的损失函数，实际应用了wing loss而不加辅助信息对网络进行训练：
$$
wingloss(x) = 
\begin{cases} 
wln(1+|x|/\epsilon),  & \text{if }|x|＜w\text{ } \\
|x| - C, & \text{otherwise }\text{}
\end{cases}
$$

#### 其他 ####

为了减小计算量，实际训练过程中没有使用PFLD框架的辅助子网络，也就是没有添加姿态角信息进行训练。

## 运行环境 ##

* Ubuntu 18.04

* Python 3.6

* Pytorch 1.9

* CUDA 10.1

* 额外需要tqdm、tensoeboard，可以使用pip install <package> 安装

* 数据集：放入在data文件夹中，需自行分配训练集（train）、验证集（validation）和测试集（predict），并将图片数据和标签数据分别放入在Images文件夹和Annotations文件夹内，如下图：

  data
  ├── predict
  │   ├── Annotations
  │   └── Images
  ├── train
  │   ├── Annotations
  │   └── Images
  └── validation
      ├── Annotations
      └── Images

  注：predict下的Annotations文件夹内可以不存放标签数据，即对测试的数据仅仅进行预测，而不进行预测与真实之间的对比，后面有说明。

## Train 训练 ##

如果你将数据集存放在其他路径下，你需要修改程序根路径下的**config.py**文件，将其中的**cfg.ROOT_TRAIN_PATH**等进行修改。

如需要有其他修改，可以修改**config.py** 文件内的epoch、lr等等信息。

训练过程只需要运行**train.py** 文件，即在程序根路径下运行 

```shell
python train.py
```

训练过程中，会自动将训练集和验证集进行训练，其中验证集并不对权重有所影响。训练结束后，训练过程中的信息会存储在**./checkpoint/log/train.log**文件中，可查看**train.log**文件，会记录每次epoch所产生的训练集loss、验证集loss和验证集RMSE。同时，训练过程中的这些信息会使用tensorboard工具将数据存放在**./checkpoint/log/** 文件夹下，即 使用如下命令（在程序根路径下）：

```shell
tensorboard --logdir=checkpoint/log
```

就可以在网页上查看数据图表。

训练后的权重文件存放在**./checkpoint/weight/pfld_ultralight_final.pth** 

## Test 测试 ##

测试过程如训练过程，只需要运行**test.py** 文件，即在程序根路径下运行：

```shell
python test.py
```

测试数据集需要将图片文件存放在**./data/predict/Images/** 文件夹中，而标签文件可选择是否存放在**./data/predict/Annotations** 文件夹中，下面进行说明：

测试过程会产生过程信息：**time**（每张图片使用的时间）、**RMSE_112**（计算图片resize成112大小之后的RMSE）和**RMSE_basic**（计算原始图片大小时的RMSE），存放在**./checkpoint/predict_log/** 文件夹下，使用tensorboard工具即可查看，即在程序根路径下运行：（若没有标签文件，则只会产生time信息）

```shell
tensorboard --logdir=checkpoint/predict_log
```

测试结果存放在**./result/** 文件夹下：

图片信息存放在**Images**文件夹下，文件名字按照训练顺序排列，对应了在tensorboard工具显示下的Loss和RMSE的横轴。每张图片中，**红点代表模型预测点，绿点代表模型真实点** 。（若没有标签文件，则没有Loss和RMSE的信息，图片上也没有绿点显示真实点）

标签信息存放在**Landmark_basic** 文件夹下，分别以txt为后缀的文件存储，文件名字是进行预测的图片的名字，即**每个标签文件与每张图片名字对应** 。

运行完test.py文件后，会在终端打印出每张图片平均的运行时间，RMSE_112和RMSE_basic的平均值。（若没有标签文件，则只会显示平均时间）

注：test.py文件自动选择**./checkpoint/weight/pfld_ultralight_final.pth** 的权重文件。

