# import csv
# import queue
#
# import pandas as pd
# import torch
#
# # from data_loader import create_dataset_loader
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
#
# from hyper_parameter import top_k
#
# class Retrival:
#     def __init__(self):
#         self.inputTotalVector = None
#
#     def calc(self, vector):
#         return loss_func(self.inputTotalVector,vector)
#
#
# def get_device():
#     return 'cuda' if torch.cuda.is_available() else 'cpu'
#
#
# def loss_func(p, z):
#     z = z.detach().clone()
#     p = torch.nn.functional.normalize(p, p=2, dim=1)
#     z = torch.nn.functional.normalize(z, p=2, dim=1)
#     return -torch.mean(torch.sum(p * z, dim=1))
#
#
# # def train_step(ds_one, ds_two, f, h):
# #
# #     z1, z2 = f(ds_one), f(ds_two)
# #     p1, p2 = h(z1), h(z2)
# #
# #     loss = loss_func(p1, z2) / 2 + loss_func(p2, z1) / 2
# #
# #     return loss.item()  # Convert the PyTorch scalar tensor to a Python float
#
#
# if __name__ == "__main__":
#
#     # retrival = Retrival()
#     # priority_queue = queue.PriorityQueue()
#     #
#     # # 加载CNN模型
#     projection = torch.load("cnn/projection.pth", map_location=torch.device(get_device()))
#     predictor = torch.load("cnn/prediction.pth", map_location=torch.device(get_device()))
#
#     # #TODO:加载GNN模型
#     #
#     #
#     # # #TODO:加载数据:从input里加载用来检索的可视化图片
#     # inputLoader = DataLoader(InputDataset, batch_size=1)
#     # for i in inputDataset.__len__():
#     #     inputSVG, inputPNG = inputDataset.__getitem__(i)
#     #     # 计算视觉信息嵌入向量
#     #     inputCNNVector = projection(inputPNG)
#     #     inputCNNVector = predictor(inputCNNVector)
#     # #     #TODO:计算结构信息嵌入向量
#     # #     # inputGNNVector =
#     # #     # 拼接为总的嵌入向量
#     # #     # retrival.inputTotalVector = torch.cat(inputCNNVector, inputGNNVector)
#     #
#     # # 要保存的列表
#     # my_list = [1, 2, 3, 4, 5]
#     #
#     # # 指定 CSV 文件名
#     # csv_file_path = 'data.csv'
#     #
#     # # 将列表保存为 CSV 文件
#     # with open(csv_file_path, 'w', newline='') as csvfile:
#     #     # 创建 CSV writer 对象
#     #     csv_writer = csv.writer(csvfile)
#     #
#     #     # 写入列表
#     #     csv_writer.writerow(my_list)
#     #
#     # # TODO:从预先存好的数据集中依次读取可视化图像对应的嵌入向量，计算内积，将序号和内积存入优先队列？
#     # # TODO:将对应的SVG复制到对应文件夹中
#     # import csv
#     #
#     # # CSV 文件路径
#     # csv_file_path = 'data.csv'
#     # # 输入SVG图像路径
#     # svg_file_path = './input/input.svg'
#     # # 读取到的有序字典（数据集的图序号和向量键值对）
#     # # data_dict = {'Index': 0, 'Vector': np.array([1, 2, 3])}
#     #
#     # # 读取SVG的嵌入向量库的CSV文件，并将其转换为列表
#     # with open(csv_file_path, 'r') as csvfile:
#     #     # 创建 CSV reader 对象
#     #     csv_reader = csv.reader(csvfile)
#     #
#     #     # # 从 CSV 文件中读取数据并转换为列表
#     #     # loaded_list = list(csv_reader)
#     #     df = pd.read_csv(csv_file_path)
#     #
#     # loaded_list = loaded_list[0]
#     # print(loaded_list)
#     #
#     # # with open(csv_file_path, 'r') as csvfile:
#     # #     # 创建 CSV reader 对象
#     # #     csv_reader = csv.reader(csvfile)
#     # #
#     # #     # # 从 CSV 文件中读取数据并转换为列表
#     # #     # loaded_list = list(csv_reader)
#     # #     csv_writer.writerow(my_ordered_dict.keys())
#     # #
#     # # loaded_list = loaded_list[0]
#     # # print(loaded_list)
#     #
#     # # dataLoader = DataLoader(TestDataset, batch_size=1)
#     # # for idx, (targetSVG, targetPNG) in enumerate(dataLoader):
#     # #     # 计算视觉信息嵌入向量
#     # cnnVector = projection(inputPNG)
#     # cnnVector = predictor(cnnVector)
#     # #     #TODO：计算结构信息嵌入向量
#     # #     # gnnVector =
#     # #
#     # #     # 拼接为总的嵌入向量
#     # #     # totalVector = torch.cat(cnnVector,gnnVector)
#     # #
#     # #     # 计算相似度
#     # #     # loss = retrival.calc(totalVector)
#     # #
#     # #     # 将元组放入优先队列里
#     # #     # priority_queue.put((loss,idx))
#     # #
#     # # # 将优先队列的前k个图片复制到对应位置并重命名
#     # # for i in range(top_k):
#     # #     if priority_queue.empty():
#     # #         break
#     # #     else:
#     # #         move(priority_queue.get(i))
#
#
#
#
