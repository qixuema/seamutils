import networkx as nx
import numpy as np


def build_graph(adjs):

    graph = nx.Graph()
    graph.add_edges_from(adjs)
    
    connected_components = nx.connected_components(graph)

    # 对每个子连通量生成对应的子图
    subgraphs = []
    for component in connected_components:
        subgraph = graph.subgraph(component).copy()
        subgraphs.append(subgraph)

    return subgraphs

# filter out the subgraphs that volume is too small
def filter_subgraphs_by_bbox(subgraphs, vertices, filter_type='volume', threshold=1e-6):
    reserved_subgraphs = []

    all_valid_nodes = []
    for sg in subgraphs:
        nodes = list(sg.nodes())
        points = vertices[nodes]
        bbox = np.array([np.min(points, axis=0), np.max(points, axis=0)])
        bbox_size = bbox[1] - bbox[0]
        
        if filter_type == 'volume':
            bbox_volume = np.prod(bbox_size)

            if bbox_volume >= threshold:
                reserved_subgraphs.append(sg)

                all_valid_nodes += nodes

        elif filter_type == 'length':
            max_edge_length = np.max(bbox_size)

            if max_edge_length >= threshold:
                reserved_subgraphs.append(sg)

                all_valid_nodes += nodes
        
        else:
            raise ValueError("Invalid filter type")


    return reserved_subgraphs, all_valid_nodes

