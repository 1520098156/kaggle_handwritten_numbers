# kaggle_handwritten_numbers
## 任务目标
运用CNN的LeNet5模型实现对kaggle手写数字的识别。  
![image](https://user-images.githubusercontent.com/52622948/159292437-a1a67759-fd54-4d2c-865f-d25aee0d55a0.png)  
## 任务流程
- 包的导入
- 设置超参数
- 预处理数据
- 构建LeNet5模型
- 训练LeNet5模型
- 验证LeNet5模型
- 在kaggle测试模型
## 包的导入
```
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from torch import nn, optim
import data_processing
import lenet5
import csv
import torch
```
我们使用csv的包来提取储存于csv格式文件中的数据。  
使用numpy来做数据的初步处理。  
使用pytorch作为构建cnn的工具包。  
## 设置超参数
在机器学习的上下文中，超参数是在开始学习过程之前设置值的参数。 相反，其他参数的值通过训练得出。  
```
num_epochs = 25 # 训练的总循环周期
batch_size = 50 # 一个撮（批次）的大小，64张图片
lr = 0.0001
```
我们知道这次的数据集的手写数字图片数据尺寸是28\*28\*1的。  
0到9共计10个数字，因此我们需要10个分类标签。
我们设定设定25个epoch，这意味着我们将会把训练集循环训练25次。  
我们设定一个batch训练50张图片。
## 预处理数据
```
def read_train_data():
    data = []
    csv_reader = csv.reader(open('digit-recognizer/train.csv'))
    next(csv_reader)
    for row in csv_reader:
        label = int(row[0])
        pic_in_row = [float(x) for x in row[1:]]
        pic = np.array(pic_in_row).reshape(1, 28, 28)
        pic = torch.from_numpy(pic).to(torch.float32)
        temp = (pic, label)
        data.append(temp)
    return data
```
这个方法使用csv的包读取数据，并将条状的（1\*784）reshape成1\*28\*28的tensor形式，将其和标签合并成一个元组temp放入名为data的list。最后返回预处理好的训练集。
```
def read_test_data():
    data = []
    csv_reader = csv.reader(open('digit-recognizer/test.csv'))
    next(csv_reader)
    for row in csv_reader:
        pic_in_row = [float(x) for x in row]
        pic = np.array(pic_in_row).reshape(1, 28, 28)
        pic = torch.from_numpy(pic).to(torch.float32)
        data.append(pic)
    return data
```
从csv文件中读取测试集并返回。
```
    train_dataset = data_processing.read_train_data()
    test_dataset = data_processing.read_test_data()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset[0:30000], batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset[30000:], batch_size=batch_size, shuffle=True)
    test_loader = torch.torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```
封装训练集验证集和测试集。读取数据集后使用DataLoader方法进行封装数据，利用多进程来加速batch data的处理，使用yield来使用有限的内存。DataLoader是一个高效，简洁，直观的网络输入数据结构，便于使用和扩展。80%的数据用来作为训练集，20%的数据用来作为测试集。
## 构建LeNet5模型
```
from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 235),
            nn.ReLU(),
            nn.Linear(235, 165),
            nn.ReLU(),
            nn.Linear(165, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

```
class LeNet5读入nn.Module参数。nn.Module 初始化了一系列重要的成员变量。这些变量初始化了在模块 forward、 backward 和权重加载等时候会被调用的的 hooks，也定义了 parameters 和 buffers。\_\_init\_\_函数继承父类。初始化两层卷积层。  
第一层和第二层卷积层先进行卷积操作，使用5\*5的感受野，移动步长是1，第一层使用6个卷积层，产生6个channel，而第二层使用16个卷积层，产生16个channel。我们希望使用padding为'VALID',根据公式算出padding为2层。接着让产生的数据通过激活函数（ReLU）,再进行归一化处理和dropout处理防止训练模型过拟合，最后在经过一次最大值池化层，减少最后连接层的中的参数数量。  
设置全三层连接层，三层的思泽分别为的size为(16 * 7 * 7, 235),(235, 165)和(165, 10)。
![image](https://user-images.githubusercontent.com/52622948/159292744-1dadb972-12e4-407f-a9c7-e9090ebda5a2.png)  
前馈步骤经过两个卷积层再经过一个全连接层后得到输出。
## 训练LeNet5模型
```
net = lenet5.LeNet5()
net = net.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
```
先进行实例化。并将模型保存在GPU。  
损失函数使用交叉熵损失函数。交叉熵损失函数是一个平滑函数，其本质是信息理论（information theory）中的交叉熵（cross entropy）在分类问题中的应用。  
优化器使用Adam优化算法，Adam 是一种可以替代传统随机梯度下降过程的一阶优化算法，它能基于训练数据迭代地更新神经网络权重，学习率等于0.0001。  
```
plot_loss = []
    plot_auc = []
    x = np.array(range(15000))

    for epoch in range(num_epochs):
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        for batch_index, (data, label) in enumerate(train_loader):
            data = data.to('cuda')
            label = label.to('cuda')

            outputs = net(data)
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)  # 更新测试图片的数量
            correct += (predicted == label).sum()  # 更新正确分类的图片的数量
            if batch_index % 100 == 0 and batch_index != 0:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch, (batch_index + 1 + epoch * len(train_loader)), sum_loss / (batch_index + 1),
                         100. * correct / total))
            plot_loss.append(sum_loss / (batch_index + 1))
            plot_auc.append(100. * correct.cpu() / total)

    plt.title('training_loss')
    plt.xlabel('batch_index')
    plt.ylabel('loss')
    plt.plot(x, np.array(plot_loss), 'r-.')
    plt.show()

    plt.title('training_accuracy')
    plt.xlabel('batch_index')
    plt.ylabel('accuracy')
    plt.plot(x, np.array(plot_auc), 'g-.')
    plt.show()

    print('training is over')
```
进行50个epoch,每个batch有50张图片被用来训练。  
先进行前馈，计算损失函数，别忘记清零上一个batch的梯度，再根据损失函数得到每个参数的梯度值，最后再通过梯度下降执行一次梯度更新。  
可以通过matplotlib.pyplot观察损失函数逐步下降趋于稳定，准确率逐步上升接近100%。
![image](https://user-images.githubusercontent.com/52622948/162576277-84fb1a4f-10b4-42db-903d-b023d32c927b.png)
![image](https://user-images.githubusercontent.com/52622948/162576287-54959e9f-81b9-4911-bea0-0880a971d8cb.png)
## 验证LeNet5模型
```
    for batch_index, (data, label) in enumerate(validation_loader):
        data = data.to('cuda')
        label = label.to('cuda')

        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)  # 更新测试图片的数量
        correct += (predicted == label).sum()  # 更新正确分类的图片的数量
        if batch_index % 100 == 0 and batch_index != 0:
            print('[batch = %d] Loss: %.03f | Acc: %.3f%% '
                  % (batch_index, sum_loss / (batch_index + 1),
                     100. * correct / total))
```
计算验证集准确率，每100个batch输出一次。  
结果如下：  
![image](https://user-images.githubusercontent.com/52622948/162576392-b75d0d5e-2765-4573-9f5e-f09ceefe8b9e.png)
## 在kaggle测试模型
```
test_prediction = []

    for batch_index, data in enumerate(test_loader):
        data = data.to('cuda')
        label = label.to('cuda')

        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)
        test_prediction += predicted.tolist()

    data_processing.write_test_prediction(test_prediction)
```
运用已有模型，对测试集进行预测并将预测结果存入csv文件。   
将预测结果提交到kaggle，得到成绩如下：  
![image](https://user-images.githubusercontent.com/52622948/162576642-7ab4897b-d70d-43cd-8b7f-cec6664baedf.png)


## 总结
完整代码：   

```
from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 235),
            nn.ReLU(),
            nn.Linear(235, 165),
            nn.ReLU(),
            nn.Linear(165, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

```
 
```
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from torch import nn, optim

import data_processing
import lenet5

num_epochs = 25
batch_size = 50
lr = 0.0001

net = lenet5.LeNet5()
net = net.to('cuda')

if __name__ == '__main__':
    train_dataset = data_processing.read_train_data()
    test_dataset = data_processing.read_test_data()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset[0:30000], batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset[30000:], batch_size=batch_size, shuffle=True)
    test_loader = torch.torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    plot_loss = []
    plot_auc = []
    x = np.array(range(15000))

    for epoch in range(num_epochs):
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        for batch_index, (data, label) in enumerate(train_loader):
            data = data.to('cuda')
            label = label.to('cuda')

            outputs = net(data)
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)  # 更新测试图片的数量
            correct += (predicted == label).sum()  # 更新正确分类的图片的数量
            if batch_index % 100 == 0 and batch_index != 0:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch, (batch_index + 1 + epoch * len(train_loader)), sum_loss / (batch_index + 1),
                         100. * correct / total))
            plot_loss.append(sum_loss / (batch_index + 1))
            plot_auc.append(100. * correct.cpu() / total)

    plt.title('training_loss')
    plt.xlabel('batch_index')
    plt.ylabel('loss')
    plt.plot(x, np.array(plot_loss), 'r-.')
    plt.show()

    plt.title('training_accuracy')
    plt.xlabel('batch_index')
    plt.ylabel('accuracy')
    plt.plot(x, np.array(plot_auc), 'g-.')
    plt.show()

    print('training is over')

    for batch_index, (data, label) in enumerate(validation_loader):
        data = data.to('cuda')
        label = label.to('cuda')

        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)  # 更新测试图片的数量
        correct += (predicted == label).sum()  # 更新正确分类的图片的数量
        if batch_index % 100 == 0 and batch_index != 0:
            print('[batch = %d] Loss: %.03f | Acc: %.3f%% '
                  % (batch_index, sum_loss / (batch_index + 1),
                     100. * correct / total))

    test_prediction = []

    for batch_index, data in enumerate(test_loader):
        data = data.to('cuda')
        label = label.to('cuda')

        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)
        test_prediction += predicted.tolist()

    data_processing.write_test_prediction(test_prediction)

```

```
import csv

import numpy as np
import torch

submission_path = 'submission.csv'


def read_train_data():
    data = []
    csv_reader = csv.reader(open('digit-recognizer/train.csv'))
    next(csv_reader)
    for row in csv_reader:
        label = int(row[0])
        pic_in_row = [float(x) for x in row[1:]]
        pic = np.array(pic_in_row).reshape(1, 28, 28)
        pic = torch.from_numpy(pic).to(torch.float32)
        temp = (pic, label)
        data.append(temp)
    return data


def read_test_data():
    data = []
    csv_reader = csv.reader(open('digit-recognizer/test.csv'))
    next(csv_reader)
    for row in csv_reader:
        pic_in_row = [float(x) for x in row]
        pic = np.array(pic_in_row).reshape(1, 28, 28)
        pic = torch.from_numpy(pic).to(torch.float32)
        data.append(pic)
    return data


def write_test_prediction(prediction):
    with open(submission_path, 'w', newline='') as file:
        csv_write = csv.writer(file)
        header = ['ImageId', 'Label']
        csv_write.writerow(header)
        for i in range(28000):
            row = [i + 1, prediction[i]]
            csv_write.writerow(row)

```

从结果来开，测试集的准确率得到了95\%以上。成功完成了对于手写数字数据集的卷积神经网络分类识别。





