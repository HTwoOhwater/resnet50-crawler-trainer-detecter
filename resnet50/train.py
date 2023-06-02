import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import time
import torchvision.datasets as datasets
import datetime
import os

class MakeTrainer:
    def __init__(self, epochs: int, learning_rate: float, num_classes: int, train_path: str, batch_size=64, test_model=True):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.test_model = test_model
        self.num_classes = num_classes
        self.train_path = train_path
        self.path = "resnet50/models/"
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, self.num_classes)
        self.model.cpu()
        # 定义转换器
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # 定义训练加载器
        self.train_dataset = datasets.ImageFolder(root=self.train_path, transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        # 定义测试加载器
        self.test_dataset = datasets.ImageFolder(root=self.train_path, transform=self.transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False)
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        # 定义优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def start(self):
        def showtime(start, end):
            time_cal = end - start
            result = ""
            second = int(time_cal % 60)
            time_cal = time_cal // 60
            minute = int(time_cal % 60)
            hour = int(time_cal // 60)
            if hour > 0:
                result = result + str(hour) + "h"
            if minute > 0:
                result = result + str(minute) + "min"
            if second > 0:
                result = result + str(second) + "s"
            return result

        if self.test_model:
            DingZhen = Image.open("原皮丁真.jpg")
            DingZhen = self.transform(DingZhen)
            plt.imshow(DingZhen.permute(1, 2, 0))
            plt.show()
            DingZhen = DingZhen.unsqueeze(0)
            print(DingZhen.shape)
            print(torch.argmax(self.model(DingZhen)))
            print("检测通过，请正常使用。")
            print("一验丁真，鉴定为一遍就过！")
            print("丁真祝您模型训练顺利！")

        start_time = time.time()
        # 获取当前日期时间
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path += current_time + "/"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images.unsqueeze(0)
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                current_time = time.time()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Total time: '.format(epoch + 1, self.epochs, i + 1,
                                                                                       len(self.train_loader), loss.item()),
                      end="")
                print(showtime(start_time, current_time))

        # ----------------------------------------------------------------------------------------------------------------------


        # 构造文件名
        filename = self.path + "/model.pth"
        # 保存模型
        torch.save(self.model.state_dict(), filename)
        file = open(self.path + "keys_to_tensors.txt", mode="a+")
        file.write(str(self.train_dataset.class_to_idx))
        file.close()
        file = open(self.path + "keys.txt", mode="a+")
        file.write(str(self.train_dataset.classes))
        file.close()
        # 返回字典值
        return self.train_dataset.classes

    def change_epoch(self, epochs: int):
        self.epochs = epochs

    def change_learning_rate(self, learning_rate: float):
        self.learning_rate =learning_rate

    def change_num_classes(self, num_classes: int):
        self.num_classes = num_classes

    def change_train_path(self, train_path):
        self.train_path = train_path

    def change_batch_size(self, batch_size):
        self.batch_size = batch_size

    def change_test_model(self, test_model):
        self.test_model = test_model