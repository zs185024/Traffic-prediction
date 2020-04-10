#-*- coding:utf-8 -*-
import numpy as np
# simport matplotlib.pyplot as plt
# import pandas as pd
from torch import nn
import torch
from torch.autograd import Variable

"""
对上面的package进行整理
numpy是一个张量包
matplotlib.pyplot主要是用来生成图片和列表的
pandas是一个用于数据分析的包
torch.nn是神经网络包
"""
"""
STEP1 怎么表示出相应的预测准确度
STEP2 
"""
# 导入数据并可视化
# data_csv = pd.read_csv('W_data.csv',usecols=[0])
# data_csv = pd.read_csv('W_data_oneday.csv',usecols=[0])
data_csv1 = np.loadtxt("W_data.csv", delimiter=" ")
# plt.plot(data_csv)
# plt.show()
# print(type(data_csv))
# plt.plot(data_csv)

# 数据预处理(去除无效数据)
dataset = data_csv1.astype('float32') # è½¬æ¢æ•°æ®ç±»åž‹ï¼Œå…¨éƒ¨è½¬æ¢æˆfloat32ä½
data_csv2 = dataset[1:,np.newaxis]
# print(type(dataset))

# 归一化处理，之前的归一化处理是归一化到0-1上，现在想试一下归一化到更大范围内的情况。
max_value = np.max(dataset)
min_value = np.min(dataset)
mu = np.mean(dataset,axis=0)
std = np.std(dataset,axis=0)
scalar = max_value - min_value
# 除最大值法归一化
# dataset = list(map(lambda x : x / scalar, data_csv2))  # 归一化处理
# MinMaxScaler法
dataset = list(map(lambda x : (x - min_value) / (max_value - min_value), data_csv2))
# 均值归一化
# dataset = list(map(lambda x : (x - mu) / std, data_csv2))
# 缩放到特定区域,[0-10]
# dataset = list(map(lambda x : x / scalar * 10 , data_csv2))


BATCH_SIZE = 144
HIDDEN_SIZE = 128

# 设置数据集(考虑滑动窗口即batch_size，所以新构造的数据集长度必定为原长-batch_size
#           ，然后这边X,Y分别是输出值，输出值，根据X来求Y，因为是训练集，所以Y告诉你)
def create_dataset(dataset, loop_back = BATCH_SIZE):
    dataX, dataY = [], []
    for i in range(len(dataset) - loop_back ):
        a = dataset[i:(i+loop_back)]    # 这边的话，因为切片是不包含的末尾位的，所以a是两个数据组成
        dataX.append(a)
        dataY.append(dataset[i+loop_back])
    return np.array(dataX), np.array(dataY)


# 创建好输入输出
# data_X，data_Y的size是(142,2,1)(142,1)，三维的数据把第一个数据当作数量，后面才是行列
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_X, data_Y = create_dataset(dataset)


# 设置训练集和测试集
# 划分训练集和测试机，其中70%作为训练集，30%位测试集，用切片来实现
# train部分，Y是给喂的数据，以便调整参数，test部分Y是用来验证正确率的
# 这边的数据集是一个三维数组
# 前三个数据集是要放到网络里的，最后一个是用于验证的
train_size = int(len(data_X)*0.8)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]


# 设置LSTM模型数据及状态(格式统一化)
# 让数据集变成仅一列，且z轴为2(batch_size)的数据结构，为了和LSTM的结构匹配
# train_X: 99,1,2
# train_Y: 99,1
# test_X:  43,1,2
train_X = train_X.reshape(-1, 1, BATCH_SIZE)  # np.reshape中的-1参数表示自适应，即-1的这个位置的值有其他位置来决定
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, BATCH_SIZE)
test_Y = test_Y.reshape(-1, 1, 1)

# 将数据从numpy的array转换为torch的tensor类型
# 所以要用GPU的话，应该要放更前面的原始数据那块进行
# 这边用torch.tensor应该
# train_x, train_y, test_x 的大小分别为99 1 2， 99 1 1  43 1 2
train_x = torch.Tensor(train_X)
# print(train_x.shape)
train_y = torch.Tensor(train_Y)
# print(train_y.shape)
test_x = torch.Tensor(test_X)
test_y = torch.Tensor(test_Y)


# 建立LSTM模型
class lstm(nn.Module):
    """
    建立本题适合的LSTM数据，输入数据大小为2，隐藏层个数为4，输出大小为1，隐藏层的层数为2，
    这些内容需要数据与之匹配，根据前面的知识可知，我们的数据集需要设置成对应的大小即
    这边的话层数设置为2，一层是LSTM，另一层是简单的线形层
    """
    def __init__(self, input_size=BATCH_SIZE, hidden_size=HIDDEN_SIZE,
                 output_size=1,num_layer=2):
        super(lstm,self).__init__()
        # 当我们的LSTM网络要求输入是2，隐藏层是4，层数为2，输出为1时
        # 我们的输入输出格式是：
        # train_X: 99,1,2
        # train_Y: 99,1
        # test_X:  43,1,2
        # 也就是说第一个参数是指元素的个数，不需要和LSTM中匹配，然后后面开始要和LSTM匹配

        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer)   # 2 4 2
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size), # 线性层 4 1
        )
    # 前向传播
    # 要注意的是这边输出的几个参数的size是什么样的
    # 然后view在这边的作用是：
    def forward(self,x):
        x,_ = self.layer1(x)  #
        s,b,h = x.size()      # 99 1 4 应该是说输入的x从9912 变成了 9914
        x = x.view(s*b, h)    # 为了通过线性层，将x的格式改成了99，4 说明4是输入需要
        # print(x.shape)
        x = self.layer2(x)    # 通过线性层得到的结果是： 99 1 是线性层的作用吧 输入4 输出1
        # print(x.shape)
        x = x.view(s,b,-1)    # 这边把格式转换成：扩维了 99 1 1 和train_Y保持一致
        # print(x.shape)
        return x

# TODO CUDA
# model = lstm(BATCH_SIZE, 8, 1, 2)
model = lstm(BATCH_SIZE, HIDDEN_SIZE, 1, 2).to(device)  # 输入为2，隐藏层为4，输出为1，层数为2，这边和输入有对应


# 建立损失函数和优化器
#TODO CUDA
# criterion = nn.MSELoss()
criterion = nn.MSELoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# torch.optim.lr_scheduler.CosineAnnealingLR()

# var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
# var_y = torch.tensor(train_y, dtype=torch.float32, device=device)
#TODO CUDA
var_x = Variable(train_x).to(device)
# var_x = var_x
var_y = Variable(train_y).to(device)
# var_y = var_y
# 模型训练
for e in range(2000):
    # var_x,var_y的格式应该和train_x,y 一样，都是9912 9911
    # 前向传播
    out = model(var_x) # 这边的out应该是model下训练出来的output
    loss = criterion(out, var_y) # 这是一个内部函数，只需要把两个要比较的张量放进去就行了
    # 反向传播
    # 在反向传播的地方进行优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 100 == 0:  # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data))

# torch.save(model.state_dict(), 'I:\\file\\newpytorch\\net.pth')
# torch.save(model.state_dict(), 'I:\\file\\newpytorch\\net3435.pth')
# torch.save(model.state_dict(), 'I:\\file\\newpytorch\\net3536.pth')
# torch.save(model.state_dict(), 'net318319.pth')
torch.save(model.state_dict(), 'net408-2.pth')
# 感觉有些地方还是不太对，尤其是正确率方面
print("save!")
## net存的是第一次训练的结果。
## 3/4-3/5调参结果存到net3435中
## net基本设置：BATCH_SIZE为288(24h)，线性层就单一层Linear。
## 3435改动：BATCH_SIZE改为12(1h) ，线性层改为双Linear。s
## 3536改动：BATCH_SIZE改为72(6h)，六个小时试一下。
## 3637改动：BATCH_SIZE改为144(12h),这次应该算是最后一次对batchsize进行测试了，后续要对LSTM网络进行测试。
## 318319改动：BATCH_SIZE仍为144，HIDDEN_SIZE=32
# TODO CUDA
model1 = lstm().to(device)
# model1.load_state_dict(torch.load('I:\\file\\newpytorch\\net.pth'))
# model1.load_state_dict(torch.load('I:\\file\\newpytorch\\net3435.pth'))
# model1.load_state_dict(torch.load('I:\\file\\newpytorch\\net3536.pth'))
model1.load_state_dict(torch.load('net408-2.pth'))
model1.eval()
print('load successfully!')
# 模型预测
# 不知道能不能直接通过调model的类型来实现测试模式的转换 ，这边还要再看下内容 。
# model = model.eval()
# model = model.eval().to(device) # 转换成测试模式
# 我们考虑的test_x放进去 然后进行结果预测
#TODO CUDA
var_test_x = Variable(test_x).to(device)
var_test_y = Variable(test_y).to(device)
# var_test_x = Variable(test_x)
# var_test_y = Variable(test_y)

# data_X = data_X.reshape(-1, 1, BATCH_SIZE)
# data_X = torch.Tensor(data_X)
# var_data = Variable(data_X)     # variable相当于是一个装tensor的存储空间，应该有其他默认参数的
#                                 # variable有一个非常大的作用，即反向误差传递的时候可以比tensor要快非常多
# TODO CUDA
pred_test = model1(var_test_x).to(device) # 测试集的预测结果
# # 改变输出的格式，即从variable形式转换为numpy or tensor类型
# pred_test = pred_test.view(-1).data.numpy()
loss1 = criterion(pred_test, var_test_y)
print(var_test_y.size())
print(len(var_test_y))
print('Loss: {:.5f}'.format(loss1.data))
running_correct = 0
# wucha = float(20) / scalar
for i in range(var_test_y.size(0)):
    if (abs((pred_test[i] - var_test_y[i])/var_test_y[i])< 0.05):
        running_correct += 1

print(running_correct)


# 6428/9617=66.84(3435--B=12,H=12)
# 6649/9605=69.2(3536--B=72,H=12)
# 6706/9591=69.9(3637--B=144,H=12)
# 6724/9591=70.1407--B=144,H=12,lr=0.005)
# 6045/9562==63.21(net--B=288,H=12)
# 6586/9591=68.6(310311--B=144.H=8)
# 6647/9591=69.3(310311-2--B=144,H=16)
# 5724/9591=59.7(318319--B=144ï¼ŒH=32)
# 6657/9594=69.4(319-1--B=128ï¼ŒH=32)
# 6558/9594=68.4(319-2--B=128ï¼ŒH=16)
# 6537/9594=68.1(320-1--B=128ï¼ŒH=12)
# 6639/9607=69.1(331-1--B=64,H=32)
# 6420/9568=67.8(403--B=256,H=24)
# 5636/9517=59.2(406--B=512,H=24)
# 6729/9591=70.1(408--B=144,H=128,lr=1e-3)
# 6740/9591=70.3(408-2--B=144,H=128,lr=1e-3,最大最小归一化法)


# 得从归一化的地方处理处理一下看一下。


# 修改了迭代次数，从5000降到了2000已经够应付1e-3的情况了
# 新改了的地方是归一化的地方，总感觉归一化到太小了以后，容易出现一些其他问题






























# 差0.05的话，只有8818/9562==92.21   6045/9562==63.21(net)
# 6428/9617=66.84(3435) 6649/9605=69.2   6706/9591=69.9 再试一次 然后考虑网络方面的
# 开始考虑hidden size 然后结合batch size一起变化看一下
# 0.08--89.1，0.07--84.5， 0.06--77.8， 0.05--69.9  6690/9561=69.8
# print(loss1.data)s

# 画出实际结果和预测的结果
# plt.title('Result Analysis')


# pred_T_t = pred_test[:,0]
# pred_N = pred_T_t.data.numpy()
# var_test_y_T = var_test_y[:,0]
# var_test_y_N = var_test_y_T.data.numpy()
# plt.plot(var_test_y_N, 'r', label='real')
# plt.plot(pred_N, 'b', label='pred')
# plt.savefig("test.png",dpi=300)s


# plt.xlabel('number')
# plt.ylabel('value')
# plt.show()
# plt.legend(loc='best')
# plt.plot(pred_y, 'r', label='pred')
# plot里面的alpha作用是
# plt.plot(data_y, 'b', label='real', alpha=0.3)
# plt.plot([train_size, train_size], [-1, 2], color='k', label='train | pred')
# plt.legend(loc='best')
# plt.savefig('lstm_reg.png')
# plt.pause(4)



