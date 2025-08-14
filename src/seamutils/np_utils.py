import numpy as np


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    assert pVec_Arr.shape == (3,), "Input vector must be of shape (3,)"

    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)
 
    z_c_vec = np.dot(z_mat, pVec_Arr)  # Matrix-vector multiplication
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    # Use np.isclose() to handle floating point precision issues with dot products
    dot_prod = np.dot(z_unit_Arr, pVec_Arr)
    dot_prod = np.clip(dot_prod, -1.0 + 1e-10, 1.0 - 1e-10)  # Prevent numerical instability near ±1
 
    if np.isclose(dot_prod, -1):
        qTrans_Mat = -np.eye(3, 3)
    elif np.isclose(dot_prod, 1):
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.dot(z_c_vec_mat, z_c_vec_mat) / (1 + dot_prod)
 
    # qTrans_Mat *= scale
    return qTrans_Mat

def filter_edges(edges):
    """
    edges: (N,2) int, assume is an undirected edge list
    return: kept_edges, keep_mask
    rule: remove edges that satisfy (one end has degree==1 and the other end has degree>2)
    """
    edges = np.asarray(edges)
    # discretize the nodes to [0, M) to avoid discontinuous node IDs
    nodes, inv = np.unique(edges, return_inverse=True)
    ed = inv.reshape(edges.shape)          # mapped edges
    deg = np.bincount(ed.ravel(), minlength=nodes.size)

    u, v = ed[:, 0], ed[:, 1]
    bad = ((deg[u] == 1) & (deg[v] > 2)) | ((deg[v] == 1) & (deg[u] > 2))
    keep = ~bad
    return edges[keep], keep
