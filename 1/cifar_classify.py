import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义超参数
batch_size = 256
num_epochs = 100
learning_rate = 0.0001

# 加载CIFAR-10数据集并进行预处理
def load_cifar10_data(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data():
    # 加载训练集
    train_images = np.empty([50000, 3, 32, 32], dtype=np.uint8)
    train_labels = np.empty([50000], dtype=np.int32)
    for i in range(5):
        data = load_cifar10_data('cifar/cifar-10-batches-py/data_batch_'+str(i+1))
        train_images[i*10000:(i+1)*10000, :, :, :] = data[b'data'].reshape((10000, 3, 32, 32))
        train_labels[i*10000:(i+1)*10000] = np.array(data[b'labels'])


    # 加载测试集
    test_data = load_cifar10_data('cifar/cifar-10-batches-py/test_batch')
    test_images = test_data[b'data'].reshape((10000, 3, 32, 32))
    test_labels = np.array(test_data[b'labels'])

    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = load_data()

# 转换为Tensor
train_images = torch.from_numpy(train_images).float()
train_labels = torch.from_numpy(train_labels).long()
test_images = torch.from_numpy(test_images).float()
test_labels = torch.from_numpy(test_labels).long()

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建模型和优化器
net = Net()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(0, len(train_images), batch_size):
        # 获取当前批次的图像和标签
        inputs = train_images[i:i+batch_size]
        labels = train_labels[i:i+batch_size]

        # 将图像和标签转换为PyTorch变量
        inputs = inputs.requires_grad_()
        labels = labels.requires_grad_(False)

        # 将梯度归零
        optimizer.zero_grad()

        # 前向传播
        outputs = net(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 统计损失
        running_loss += loss.item()

    # 输出训练损失
    print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss / (len(train_images) / batch_size)))

    # 在测试集上评估模型性能
    correct = 0
    total = 0
    for i in range(0, len(test_images), batch_size):
        # 获取当前批次的图像和标签
        inputs = test_images[i:i+batch_size]
        labels = test_labels[i:i+batch_size]

        # 将图像转换为PyTorch变量
        inputs = inputs.requires_grad_()

        # 前向传播
        outputs = net(inputs)

        # 预测类别
        _, predicted = torch.max(outputs.data, 1)

        # 统计正确预测的样本数
        total += labels.size(0)
        correct += (predicted == labels).sum()

    # 输出测试准确率
    print('Test Accuracy: %.2f %%' % (100.0 * correct / total))