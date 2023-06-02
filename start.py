import resnet50.predict as predict
import resnet50.train as train
import crawler.crawler as crawler
import tkinter as tk
from tkinter.filedialog import *


"""
等待加上去的功能：
1.结合起来爬虫√
2.结合起来训练√
3.结合起来预测√
4.预测可以用摄像头√
5.图形化界面
"""

"""
def selectFile():
    global img
    filepath = askopenfilename()  # 选择打开什么文件，返回文件名
    filename.set(filepath)  # 设置变量filename的值
    img = Image.open(filename.get())  # 打开图片


def outputFile():
    outputFilePath = askdirectory()  # 选择目录，返回目录名
    outputpath.set(outputFilePath)  # 设置变量outputpath的值

root = tk.Tk()
filename = tk.StringVar()
outputpath = tk.StringVar()


# 构建“选择文件”这一行的标签、输入框以及启动按钮，同时我们希望当用户选择图片之后能够显示原图的基本信息
tk.Label(root, text='选择文件').grid(row=1, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=filename).grid(row=1, column=1, padx=5, pady=5)
tk.Button(root, text='打开文件', command=selectFile).grid(row=1, column=2, padx=5, pady=5)

# 构建“选择目录”这一行的标签、输入框以及启动按钮
tk.Label(root, text='选择目录').grid(row=2, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=outputpath).grid(row=2, column=1, padx=5, pady=5)
tk.Button(root, text='点击选择', command=outputFile).grid(row=2, column=2, padx=5, pady=5)

root.mainloop()
"""

"""
def scan():
    print(type(Acrawler))

def pa():
    global Acrawler
    Acrawler.start()

def open_settings_crawler():
    def save_parameters():
        global Acrawler
        # 获取文本框中的参数值
        parameter1_value = parameter1_entry.get()
        parameter2_value = parameter2_entry.get()
        parameter3_value = parameter3_entry.get()
        parameter4_value = parameter4_entry.get()
        # 打印参数值，你可以在这里进行其他操作，比如保存参数到文件等
        Acrawler = crawler.MakeCrawler(parameter1_value, parameter2_value, int(parameter3_value), parameter4_value)
        print("Parameter 1:", parameter1_value)
        print("Parameter 2:", parameter2_value)
        print("Parameter 3:", parameter3_value)
        print("Parameter 4:", parameter4_value)

    settings_window = tk.Toplevel(root)
    settings_window.geometry("300x200")

    # 创建文本框和标签，用于输入和显示参数值
    parameter1_label = tk.Label(settings_window, text="爬取图片的名称")
    parameter1_label.pack()
    parameter1_entry = tk.Entry(settings_window)
    parameter1_entry.pack()

    parameter2_label = tk.Label(settings_window, text="爬取的图片放置路径")
    parameter2_label.pack()
    parameter2_entry = tk.Entry(settings_window)
    parameter2_entry.pack()

    parameter3_label = tk.Label(settings_window, text="爬取的图片数目")
    parameter3_label.pack()
    parameter3_entry = tk.Entry(settings_window)
    parameter3_entry.pack()

    parameter4_label = tk.Label(settings_window, text="已经爬取的图片数目")
    parameter4_label.pack()
    parameter4_entry = tk.Entry(settings_window)
    parameter4_entry.pack()

    # 创建保存按钮
    save_button = tk.Button(settings_window, text="保存参数", command=save_parameters)
    save_button.pack()

    settings_window.mainloop()


Acrawler = None

root = tk.Tk()
root.geometry("400x600")
button = tk.Button(root, text="打开爬虫设置", command=open_settings_crawler)
button.pack()
button_1 = tk.Button(root, text="爬！", command=pa)
button_1.pack()
button_2 = tk.Button(root, text="看看你的类对不对", command=scan)
button_2.pack()
root.mainloop()

"""


predicter = predict.MakePredictor(model_path="./resnet50/models/猫狗识别.pth", num_classes=2)
predicter.predict_camera(output_list=["猫猫", "狗狗"])

