import torchvision.models as models
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pylab import mpl
import cv2


mpl.rcParams["font.sans-serif"] = ["SimHei"]


class MakePredictor:
    def __init__(self, model_path: str, num_classes: int, output_list=[]):
        # 模型路径
        self.model_path = model_path
        # 模型输出维度
        self.num_classes = num_classes
        # 这里写了一个转换器，将输入的图片转化为224*224的RGB图片
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # 输出对应关系列表
        self.output_list = output_list
        # 是否输出原始维度信息
        if not self.output_list:
            self.raw_output_mode = True
        else:
            self.raw_output_mode = False
        # 定义模型
        self.model = models.resnet50(num_classes=self.num_classes, weights=None)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def change_model_path(self, path: str):
        self.model_path = path

    def change_output_list(self, output_list: list):
        self.output_list = output_list
        if not self.output_list:
            self.raw_output_mode = True
        else:
            self.raw_output_mode = False

    def change_num_classes(self, num_classes: int):
        self.num_classes = num_classes

    def change_input_mode(self, input_mode: str):
        self.input_mode = input_mode

    def predict_image(self, image_path: str):
        image = plt.imread(image_path)
        image_show = image
        image = self.transform(image)
        output = []

        # 进行预测
        image = image.unsqueeze(0)
        result = self.model(image)
        value, index = torch.max(result, dim=1)

        if not self.output_list:
            print(result, value, index)
            output.append([result, value, index])
        else:
            plt.imshow(image_show)
            if value > 2:
                type_detected = self.output_list[index]
                text = "识别到了: {}".format(type_detected)
                # 在图表中添加文本注释
                plt.text(10, 10, text, bbox=dict(facecolor='red', alpha=0.8))
            else:
                text = "未能识别到目标"
                # 在图表中添加文本注释
                plt.text(10, 10, text, bbox=dict(facecolor='green', alpha=0.8))
        plt.show()
        return output

    def predict_filefolder(self, folder_path: str):
        test_dataset = datasets.ImageFolder(root="D:/User/documents/pycharm/resnet+crawler+autotrain/test/")
        output = []
        for image, label in test_dataset:
            image_show = image
            image = self.transform(image)

            # 进行预测
            image = image.unsqueeze(0)
            result = self.model(image)
            value, index = torch.max(result, dim=1)

            if not self.output_list:
                print(result, value, index)
                output.append([result, value, index])
            else:
                plt.imshow(image_show)
                if value > 2:
                    type_detected = self.output_list[index]
                    text = "识别到了: {}".format(type_detected)
                    # 在图表中添加文本注释
                    plt.text(10, 10, text, bbox=dict(facecolor='red', alpha=0.8))
                else:
                    text = "未能识别到目标"
                    # 在图表中添加文本注释
                    plt.text(10, 10, text, bbox=dict(facecolor='green', alpha=0.8))
            plt.show()
        return output

    def predict_camera(self, output_list=[]):
        # 打开摄像头
        cap = cv2.VideoCapture(0)  # 参数0表示使用默认摄像头，如果有多个摄像头可以尝试不同的参数
        # 检查摄像头是否成功打开
        if not cap.isOpened():
            print("无法打开摄像头")
            exit()
        # 循环读取和显示图像帧
        while True:
            # 从摄像头读取图像帧
            ret, frame = cap.read()
            # 检查图像帧是否成功读取
            if not ret:
                print("无法获取图像帧")
                break
            # 进行预测
            image = frame
            image = self.transform(image)
            image = image.unsqueeze(0)
            result = self.model(image)
            value, index = torch.max(result, dim=1)
            if value > 2:
                text = output_list[index]
                cv2.putText(frame, text, (10, 50), (0, 255, 0), cv2.FontFace("UTF8"), 50)
            else:
                text = "未检测到对象"
                cv2.putText(frame, text, (10, 50), (0, 0, 255), cv2.FontFace("UTF8"), 50)

            cv2.imshow("Frame", frame)
            # 按下 'q' 键退出循环
            if cv2.waitKey(1) == ord('q'):
                break

        # 释放摄像头和关闭窗口
        cap.release()
        cv2.destroyAllWindows()
