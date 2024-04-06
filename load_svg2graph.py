import torch
from torch_geometric.data import Data
import os
from xml.etree import ElementTree


class Graph:
    def __init__(self):
        self.id_dict = {}
        self.count = {'node': 0, 'edge': 0}
        self.v_list = []
        self.e_list = []


class Vertex:
    def __init__(self,
                 tag: str,
                 in_graph: Graph,
                 is_root: bool = False,
                 is_def: bool = False,
                 is_clip: bool = False,
                 is_entity: bool = False,
                 is_use: bool = False,
                 be_used: bool = False,
                 has_curve: bool = False,
                 has_line: bool = False):
        self.tag = tag
        self.attr = {'is_root': False, 'is_def': False, 'is_clip': False, 'is_entity': False, 'is_use': False,
                     'be_used': False, 'has_curve': False, 'has_line': False}
        self.start = []
        self.end = []
        if is_root:
            self.attr['is_root'] = True
        if is_def:
            self.attr['is_def'] = True
        if is_clip:
            self.attr['is_clip'] = True
        if is_entity:
            self.attr['is_entity'] = True
        if is_use:
            self.attr['is_use'] = True
        if be_used:
            self.attr['be_used'] = True
        if has_curve:
            self.attr['has_curve'] = True
        if has_line:
            self.attr['has_line'] = True
        in_graph.count['node'] += 1
        in_graph.v_list.append(self)

    def to_list(self):
        attr = []
        for i in self.attr:
            attr.append(1.0 if self.attr[i] else 0.0)
        return attr


class Edge:
    def __init__(self, start: Vertex, end: Vertex, in_graph: Graph,
                 hold: bool = False,
                 be_held: bool = False,
                 use: bool = False,
                 be_used: bool = False,
                 auto_reverse: bool = False):
        self.start = start
        self.end = end
        start.start.append(self)
        end.end.append(self)
        self.attr = {'hold': False, 'be_held': False, 'use': False, 'be_used': False}
        if hold:
            self.attr['hold'] = True
            if auto_reverse:
                Edge(end, start, in_graph, be_held=True)
        if be_held:
            self.attr['be_held'] = True
            if auto_reverse:
                Edge(end, start, in_graph, hold=True)
        if use:
            self.attr['use'] = True
            if auto_reverse:
                Edge(end, start, in_graph, be_used=True)
        if be_used:
            self.attr['be_used'] = True
            if auto_reverse:
                Edge(end, start, in_graph, use=True)
        in_graph.count['edge'] += 1
        in_graph.e_list.append(self)

    def to_list(self):
        attr = []
        for i in self.attr:
            attr.append(1.0 if i[1] else 0.0)
        return attr


def svg2graph(path, y)  :
    if not os.path.isfile(path):
        raise Exception
    graph = Graph()
    root = ElementTree.parse(path).getroot()
    graph_root = parse_node_step1(root, graph)
    x, edge_index, edge_attr = parse_graph_step2(graph)
    return Data(x=torch.tensor(x, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long).t(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float), y=y)


def parse_node_step1(root: ElementTree.Element, graph: Graph, vis=True):
    node = Vertex(root.tag, graph)
    if node.tag == '{http://www.w3.org/2000/svg}svg' or node.tag == '{http://www.w3.org/2000/svg}g':
        node.attr['is_root'] = True
    elif node.tag == '{http://www.w3.org/2000/svg}defs' or node.tag == '{http://www.w3.org/2000/svg}symbol':
        node.attr['is_def'] = True
        vis = False
    elif node.tag == '{http://www.w3.org/2000/svg}clipPath':
        node.attr['is_clip'] = True
        vis = False
    elif node.tag == '{http://www.w3.org/2000/svg}rect':
        if vis:
            node.attr['is_entity'] = True
        node.attr['has_line'] = True
    elif node.tag == '{http://www.w3.org/2000/svg}path':
        if vis:
            node.attr['is_entity'] = True
        temp = root.get('d').lower()
        if temp.find('q') is not None or temp.find('t') is not None or temp.find('a') is not None or \
                temp.find('c') is not None or temp.find('s') is not None:
            node.attr['has_curve'] = True
        if temp.find('l') is not None or temp.find('h') is not None or temp.find('v') is not None:
            node.attr['has_line'] = True
    elif node.tag == '{http://www.w3.org/2000/svg}use':
        node.attr['is_use'] = True
    else:
        return None
    if root.get('id') is not None:
        graph.id_dict[root.get('id')] = node
    if root.get('{http://www.w3.org/1999/xlink}href') is not None:
        ppp = root.get('{http://www.w3.org/1999/xlink}href')
        ppp = ppp[len('#'):]
        origin = graph.id_dict.get(ppp)
        # origin = graph.id_dict.get(root.get('{http://www.w3.org/1999/xlink}href').removeprefix('#'))
        if origin is None:
            return None
        else:
            Edge(node, origin, graph, use=True, auto_reverse=True)
            origin.attr['be_used'] = True
    if root.get('clip-path') is not None:
        ppp1 = root.get('clip-path')
        ppp1 = ppp1[len('#'):]
        origin = graph.id_dict.get(ppp1)
        # origin = graph.id_dict.get(root.get('clip-path').removeprefix('#'))
        if origin is None:
            return None
        else:
            Edge(node, origin, graph, use=True, auto_reverse=True)
            origin.attr['be_used'] = True
    for i in root:
        child = parse_node_step1(i, graph, vis)
        if child is not None:
            Edge(node, child, graph, hold=True, auto_reverse=True)
    return node


def parse_graph_step2(graph: Graph):
    x = []
    edge_index = []
    edge_attr = []
    for i in graph.v_list:
        x.append(i.to_list())
    for i in graph.e_list:
        edge_index.append([graph.v_list.index(i.start), graph.v_list.index(i.end)])
        edge_attr.append(i.to_list())
    return x, edge_index, edge_attr


def process(in_path, out_path):
    if not os.path.isdir(in_path):
        raise NotADirectoryError
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for name in os.listdir(in_path):
        in_name = os.path.join(in_path, name)
        out_name = os.path.join(out_path, name)
        if os.path.isfile(in_name):
            if in_path.find('bar') >= 0:
                y = 0
            elif in_path.find('line') >= 0:
                y = 1
            elif in_path.find('pie') >= 0:
                y = 2
            elif in_path.find('scatter') >= 0:
                y = 3
            else:
                y = -1
            temp = svg2graph(in_name, y)
            print(temp)
            torch.save(temp, out_name.split(name.split('.')[-1])[0]+'pyg')
        else:
            process(in_name, out_name)
