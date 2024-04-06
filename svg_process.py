import os.path

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import load_svg2graph
import svg2img

svg_path = './dataset_svg'
graph_path = './dataset_graph'
png_path = './dataset_img'


if __name__ == '__main__':
    load_svg2graph.process(svg_path, graph_path)
    svg2img.svg2png(svg_path, png_path)
