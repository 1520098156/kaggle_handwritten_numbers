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
