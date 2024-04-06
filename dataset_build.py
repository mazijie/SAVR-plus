import csv
import os
import shutil

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# from data_loader import create_dataset_loader


class CNNDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        # if self.transform:
        #     anthor = self.transform(image)
        #     positive = self.transform(image)
        transform = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
        ])
        image = transform(image)
        return image, idx



def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def embedding(ds, f, h):

    z = f(ds)
    p = h(z)
    return p


def traverse_files(graph_path, png_path, cnn_encoder, cnn_predictor):

    # 计算图像数量
    num_files = len(os.listdir(png_path))
    shutil.copyfile("./middle/test.png", "./dataset_img/"+str(num_files)+".png")
    # assert num_files == len(os.listdir(graph_path))

    cnnDataLoader = DataLoader(CNNDataset(png_path), batch_size=16, shuffle=False, num_workers=12)
    if os.path.exists('cnn.tsv'):
        # 如果文件存在，删除文件
        os.remove('cnn.tsv')
    if os.path.exists('storage.tsv'):
        # 如果文件存在，删除文件
        os.remove('storage.tsv')
    count=0
    for i in cnnDataLoader:
        cnn_predictor.eval()
        cnn_encoder.eval()
        cnn_vector = cnn_encoder(i[0])
        cnn_vector = cnn_predictor(cnn_vector)
        cnn_vector = torch.nn.functional.normalize(cnn_vector, p=2, dim=1)
        data = cnn_vector.detach().numpy()

        with open('cnn.tsv', 'a', newline='', encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')

            # 逐行写入数据
            for row in data:
                writer.writerow(row)
                print(count)
                count+=1
        # for item in data:
        #     np.savetxt('output.tsv', item, delimiter='\t', fmt='%lf')

    os.remove("./dataset_img/"+str(num_files)+".png")
    cnntsvfile = open("cnn.tsv", 'r', newline='', encoding='utf-8')
    cnnreader = csv.reader(cnntsvfile,delimiter='\t')
    gnntsvfile = open("gnn.tsv", 'r', newline='', encoding='utf-8')
    gnnreader = csv.reader(gnntsvfile, delimiter='\t')
    with open("storage.tsv", 'a', newline='', encoding='utf-8') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        for i in range(num_files):
            cnn_vector = np.array(cnnreader.__next__())
            cnn_vector = [float(value) for value in cnn_vector]
            gnn_vector = np.array(gnnreader.__next__())
            gnn_vector = [float(value) for value in gnn_vector]
            cnn_vector = torch.tensor(cnn_vector)
            gnn_vector = torch.tensor(gnn_vector)
            gnn_vector = torch.nn.functional.normalize(gnn_vector, p=2,dim=0)
            # embeding=gnn_vector
            embeding = torch.cat((cnn_vector, gnn_vector))
            # embeding = torch.cat((cnn_vector,gnn_vector,gnn_vector,gnn_vector,gnn_vector,gnn_vector,gnn_vector,gnn_vector,gnn_vector))
            # embeding = np.concatenate((cnn_vector,gnn_vector,gnn_vector,gnn_vector,gnn_vector,gnn_vector,gnn_vector,gnn_vector,gnn_vector))
            embeding = torch.nn.functional.normalize(embeding, p=2, dim=0)
            print(embeding.shape)
            embeding= embeding.detach().numpy()
            writer.writerow(embeding)


if __name__ == "__main__":

    graph_path = "./dataset_graph"
    png_path = "./dataset_img"
    projection = torch.load("cnn/projection.pth", map_location=torch.device(get_device()))
    predictor = torch.load("cnn/prediction.pth", map_location=torch.device(get_device()))
    traverse_files(graph_path,png_path,projection,predictor)

