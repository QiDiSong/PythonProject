import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_data(startyear, endyear, path, features, rows, cols):
    '''
    :param startyear: 开始年份
    :param endyear: 结束年份
    :param path: 路径（同级文件夹下）
    :param rows: 将图片看作矩阵，行数
    :param cols:列数
    :return:返回三维列表——rows*cols行，每行有（endyear+1-startyear）个元素列表，每个列表包含这个点在该年的特征
    '''
    # 创建数据结构
    dataset  = [[] for i in range(rows * cols)]
    for i in range(rows * cols):
        dataset[i] = [[] for year in range(startyear, endyear+1)]

    # 对feature特征第year年的数据图
    for feature in features:
        for year in range(startyear, endyear + 1):
            # 读取图片
            data_path = f"./{path}/{feature}/{feature}_{year}.tif"
            # print(f"Reading image from: {data_path}")  # 输出图片路径
            img = cv2.imread(data_path, -1)  # 读取图像
            # 输出图片大小
            # if year == startyear :
            #     print(f"Image size: {img.shape[0]} x {img.shape[1]}")

            # 从图片中读取数据，假设每个像素代表一个数据点
            for i in range(rows):
                for j in range(cols):
                    dataset[i * cols + j][year - startyear].append(img[i][j])

    return dataset


def process(dataset):
    note = [0] * len(dataset)  # 初始化note数组，全部为0，用于标记是否删除某个dataset[i]

    for i in range(len(dataset)):
        if note[i] == 1:
            break
        for j in range(len(dataset[0])):
            for k in range(len(dataset[i][j])):
                if dataset[i][j][k] < 1 or dataset[i][j][k] > 365:
                    note[i] = 1
                    break

    processed_dataset = []
    for i in range(len(dataset)):
        if note[i] == 0:
            processed_dataset.append(dataset[i])

    return processed_dataset,note


# 绘图
def paint_result(note, res, K, row, col):
    '''
    note: 存储了异常点
    res: 存储了去掉异常点后的聚类结果
    K: 聚类成为K类
    目的：还原成row*col大小的聚类可视化图片
    '''
    # 定义颜色列表
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    # 定义result列表
    result = np.zeros((row, col, 3), dtype=np.uint8)  # Initialize an empty image array
    i = 0
    # 遍历note
    for r in range(row):
        for c in range(col):
            # 计算当前像素的位置
            index = r * col + c
            # 如果note == 0
            if note[index] == 0:
                # 则该点的像素记作colors[res[i]]
                result[r, c] = colors[res[i]]
                i += 1
            # 如果note == 1
            elif note[index] == 1:
                # 则该点的像素记为白色
                result[r, c] = (0, 0, 0)  # White color in RGB
            # 如果i的索引超过了res的长度，则跳出循环
            if i >= len(res):
                break
        if i >= len(res):
            break

    # 显示图像
    plt.imshow(result)
    plt.axis('off')  # 隐藏坐标轴
    # 保存图像
    plt.savefig('result_image.png', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':

    # 读取三个特征 + 预处理
    # 读取数据
    dataset = read_data(2001, 2017, "cluster_data", ["DO", "GO", "PEAK"], 435, 1226)
    # 输出基本特征
    print("数据集形状:", len(dataset), "行,", len(dataset[0]), "个时间点,", len(dataset[0][0]), "个数据特征")
    print("总计数据点数:", len(dataset) * len(dataset[0]))
    # 预处理
    processed_dataset, note = process(dataset)
    processed_dataset = np.transpose(processed_dataset, (0, 2, 1))
    np.save('processed_dataset.npy', processed_dataset)
    # loaded_dataset = np.load('processed_dataset.npy')
    print("数据集形状:", len(processed_dataset), "行,", len(processed_dataset[0]), "个时间点,", len(processed_dataset[0][0]), "个数据特征")
    print("总计数据点数:", len(processed_dataset) * len(processed_dataset[0]))
    # 数据集是三维列表，聚成五/六类


    # 跑那个网络，给出列表表示的聚类结果res，如[1,2,3, 2, 3, 1, 5,4,]这样
    # paint_result(note, res, 5, 435, 1226)


