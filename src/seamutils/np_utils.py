import numpy as np
from qixuema.np_utils import deduplicate_lines, deduplicate_faces, rotation_matrix_z

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


def get_rotaion_matrix_3d(idx):
    # idx 0, 1, 2, 3
    angles = [0, 90, 180, 270]
    angle = angles[idx]
    rot_matrix = rotation_matrix_z(angle)
    return rot_matrix

def remove_duplicate_vertices_and_lines_for_seam(mesh:dict, tolerance=0.0001):
    # 注意，在这部分的代码中，我们并没有对顶点的顺序进行排序，我们只是剔除了重复（三维空间接近）的顶点,
    vertices = mesh['vertices']

    if not np.all(np.mod(vertices, 1) == 0):
        adjusted_points = np.round(vertices / tolerance) * tolerance
    else:
        adjusted_points = vertices
    
    # 移除重复的点并创建索引映射    
    unique_points, inverse_indices = np.unique(adjusted_points, axis=0, return_inverse=True)

    mesh['vertices'] = unique_points

    # if lines are provided, update the lines' indices
    if 'lines' in mesh and mesh['lines'] is not None:
        lines = mesh['lines']

        updated_lines = inverse_indices[lines] # 更新线段索引，但是线段的顺序还是不变

        unique_lines = deduplicate_lines(updated_lines)

        valid_lines = clean_invalid_lines(unique_lines)
        
        mesh['lines'] = valid_lines

    # If faces are provided, update the faces' indices
    if 'faces' in mesh and mesh['faces'] is not None:
        faces = mesh['faces']
        updated_faces = inverse_indices[faces] # update the vtx idx in faces
                
        unique_faces = deduplicate_faces(updated_faces)
        
        unique_faces = clean_invalid_faces(unique_faces)
        
        # Update faces in the lineset
        mesh['faces'] = unique_faces
    
    if 'chains_1D' in mesh and mesh['chains_1D'] is not None:
        
        chains_1D = mesh['chains_1D']
        
        updated_chains_1D = np.where(chains_1D == -1, -1, inverse_indices[chains_1D]) # update the chains_1D indices

        mesh['chains_1D'] = updated_chains_1D
    
    return mesh

def clean_invalid_faces(faces):
    diffs = np.abs(faces[:, [0, 0, 1]] - faces[:, [1, 2, 2]]).min(axis=1)
    mask = diffs < 0.5
    return faces[~mask]

def clean_invalid_lines(lines):
    diff = np.abs(lines[:, 0] - lines[:, 1])
    return lines[np.abs(diff) >= 0.5]
