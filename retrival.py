import csv
import os
import shutil
from queue import PriorityQueue

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

import encoder
import predictor
from hyper_parameter import input_root_dir, top_k


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class InputPNGDataset(Dataset):
    def __init__(self):
        self.root_dir = input_root_dir
        self.images = os.listdir()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        png_path = os.path.join(self.root_dir, str("test")+".png")
        png = Image.open(png_path).convert("RGB")

        # if self.transform:
        #     anthor = self.transform(image)
        #     positive = self.transform(image)
        transform = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
        ])
        image = transform(png)
        return image, idx


def retrival(cnn_encoder, cnn_predictor):

    # 计算图像数量
    # num_files = len(os.listdir(graph_path))
    # assert num_files == len(os.listdir(png_path))
    cnnVector = None
    # cnnInputDataLoader = DataLoader(InputPNGDataset(), batch_size=2, shuffle=False, num_workers=1)
    # for i in cnnInputDataLoader:
    #     cnn_predictor.eval()
    #     cnn_encoder.eval()
    #     cnn_vector = cnn_encoder(i[0])
    #     cnn_vector = cnn_predictor(cnn_vector)
    #     cnn_vector = torch.nn.functional.normalize(cnn_vector, p=2, dim=1)
    #     data = cnn_vector.detach().numpy()
    #     cnnVector = data[0]
        # print(cnnVector)
    cnnfile = open("cnn.tsv", 'r', newline='', encoding='utf-8')
    reader = csv.reader(cnnfile, delimiter='\t')
    lines = list(reader)
    line = lines[-1]
    cnnVector = [float(value) for value in line]
    cnnVector = torch.tensor(cnnVector)

    # gnnVector = np.zeros(2048)
    gnnfile = open("gnn.tsv", 'r', newline='', encoding='utf-8')
    reader = csv.reader(gnnfile,delimiter='\t')
    lines = list(reader)
    line = lines[-1]
    gnnVector = [float(value) for value in line]
    gnnVector = torch.tensor(gnnVector)

    # embeding = gnnVector
    embeding = np.concatenate(
        (cnnVector, gnnVector))
    # print(embeding)
    # embeding = np.concatenate((cnnVector, gnnVector,gnnVector,gnnVector,gnnVector,gnnVector,gnnVector,gnnVector,gnnVector))

    tsvfile = open("storage.tsv", 'r', newline='', encoding='utf-8')
    reader = csv.reader(tsvfile, delimiter='\t')
    idx = 0
    embeding = torch.tensor(embeding)
    embeding = torch.nn.functional.normalize(embeding, p=2, dim=0)
    # print(torch.norm(embeding))
    priority_queue = PriorityQueue()
    for row in reader:
        # print(row)
        # break
        processed_row = [float(value) for value in row]
        # print(processed_row)
        processed_row = torch.tensor(processed_row)
        # print(torch.norm(processed_row))
        loss = -torch.sum(processed_row * embeding, dim=0)
        # loss = torch.norm(processed_row-embeding)
        # if loss>0:
        #     loss=-loss
        # print(loss)
        priority_queue.put((loss, idx))
        idx += 1
    # print(idx)
    count = 1
    # priority_queue.get()
    while not priority_queue.empty() and not count > top_k:
        loss, i = priority_queue.get()
        print(loss, i)
        source_path = "./dataset_svg/"+str(i)+".svg"
        destination_path = "./output/"+str(count)+".svg"
        shutil.copyfile(source_path, destination_path)
        source_path = "./dataset_img/" + str(i) + ".png"
        destination_path = "./output/" + str(count) + ".png"
        shutil.copyfile(source_path, destination_path)
        count+=1
        # print(loss, i)


if __name__ == "__main__":

    projection = torch.load("cnn/projection.pth", map_location=torch.device(get_device()))
    predictor = torch.load("cnn/prediction.pth", map_location=torch.device(get_device()))
    retrival(projection, predictor)
