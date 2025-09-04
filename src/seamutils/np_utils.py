import numpy as np
from qixuema.np_utils import (
    deduplicate_lines, deduplicate_faces, 
    rotation_matrix_z, boundary_vertex_indices,
    clean_invalid_faces, clean_invalid_lines,
    tolerant_lexsort, normalize_vertices, 
    rotation_matrix_x, rotation_matrix_y, rotation_matrix_z,
)
from seamutils.base import (
    sort_and_deduplicate_chains, split_and_filter_chains_1D, split_graph_into_chains,
    filter_chains, flatten_and_add_marker, ratio_of_single_edge_chain ,
)

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

def remove_duplicate_vertices_and_lines_for_seam(mesh_data:dict, tolerance=1e-6):
    # 注意，在这部分的代码中，我们并没有对顶点的顺序进行排序，我们只是剔除了重复（三维空间接近）的顶点,
    vertices = mesh_data['vertices']

    if not np.all(np.mod(vertices, 1) == 0):
        adjusted_points = np.round(vertices / tolerance) * tolerance
    else:
        adjusted_points = vertices
    
    # 移除重复的点并创建索引映射    
    unique_points, inverse_indices = np.unique(adjusted_points, axis=0, return_inverse=True)

    mesh_data['vertices'] = unique_points

    # if lines are provided, update the lines' indices
    if 'lines' in mesh_data and mesh_data['lines'] is not None:
        lines = mesh_data['lines']

        updated_lines = inverse_indices[lines] # 更新线段索引，但是线段的顺序还是不变

        unique_lines = deduplicate_lines(updated_lines)

        valid_lines = clean_invalid_lines(unique_lines)
        
        mesh_data['lines'] = valid_lines

    # If faces are provided, update the faces' indices
    if 'faces' in mesh_data and mesh_data['faces'] is not None:
        faces = mesh_data['faces']
        updated_faces = inverse_indices[faces] # update the vtx idx in faces
                
        unique_faces = deduplicate_faces(updated_faces)
        
        unique_faces = clean_invalid_faces(unique_faces)
        
        # Update faces in the lineset
        mesh_data['faces'] = unique_faces
    
    if 'chains_1D' in mesh_data and mesh_data['chains_1D'] is not None:
        
        chains_1D = mesh_data['chains_1D']
        
        updated_chains_1D = np.where(chains_1D == -1, -1, inverse_indices[chains_1D]) # update the chains_1D indices

        mesh_data['chains_1D'] = updated_chains_1D
    
    return mesh_data

def sort_vertices_and_update_indices(sample):
    """
    assume the vertices and lines are unique
    """
    
    # append a new vertex to the end of the vertices
    vertices = sample['vertices']
    faces = sample['faces']
    chains_1D = sample['chains_1D'] if 'chains_1D' in sample else None
    lines = sample['lines']
    
    # Sort vertices by z then y then x.
    # sort_vtx_inds = np.lexsort(vertices.T)
    sort_vtx_inds = tolerant_lexsort(vertices)
    
    vertices_updated = vertices[sort_vtx_inds]

    reverse_sort_vtx_inds = np.argsort(sort_vtx_inds)    

    if chains_1D is None:
        
        lines_updated = reverse_sort_vtx_inds[lines]
        
        chains = split_graph_into_chains(lines_updated.tolist())
        
        # NOTE：这里如果一条 chain 完全位于一个 surface 的内部，则过滤掉；但是这种情况可能会收到 non-manif vert 的影响，因为 non-manif vert 也可能会被认为是 boundary edge！
        boundary_vtx_idxes = boundary_vertex_indices(faces)
        
        chains = filter_chains(chains, boundary_vtx_idxes)
    else:
        updated_chains_1D = np.where(chains_1D == -1, -1, reverse_sort_vtx_inds[chains_1D]) # update the chains_1D indices
        
        chains = split_and_filter_chains_1D(updated_chains_1D)
    
    assert len(chains) > 0, "No valid chains found"
    
    chains = sort_and_deduplicate_chains(chains)
    chains_1D_dict = flatten_and_add_marker(chains)
    
    chains_1D_dict['ratio_single_edge_chain'] = ratio_of_single_edge_chain(chains)
    chains_1D_dict['chains'] = chains
    
    # Just for debug
    # random_idx = np.random.randint(10, 30)
    # chains_1D_updated = chains_1D_updated[:random_idx]
    
    if faces is not None:
        # Re-index faces and tris to re-ordered vertices.
        faces_updated = reverse_sort_vtx_inds[faces]
    else:
        faces_updated = None

    return {
        'vertices': vertices_updated,
        'faces': faces_updated,
        'chains': chains_1D_dict,
    }
    

def get_random_rotation_matrix_3d(interval_angle=90, rotation_axis=2):
    # Generate a random rotation angle between 0 and 360 degrees
    assert interval_angle > 0 and interval_angle <= 360, "interval_angle should be in (0, 360]"
    assert 360 % interval_angle == 0, "interval_angle should be a divisor of 360"
    
    assert rotation_axis in [0, 1, 2, 3], "axis should be in [0, 1, 2, 3]"
    
    random_angle = np.random.randint(0, 360 // interval_angle, size=3) * interval_angle
    
    if rotation_axis == 0:
        rot_matrix = rotation_matrix_x(random_angle[0])
    elif rotation_axis == 1:
        rot_matrix = rotation_matrix_y(random_angle[1])
    elif rotation_axis == 2:
        rot_matrix = rotation_matrix_z(random_angle[2])
    elif rotation_axis == 3:
        Rx = rotation_matrix_x(random_angle[0])
        Ry = rotation_matrix_y(random_angle[1])
        Rz = rotation_matrix_z(random_angle[2])
        rot_matrix = Rx @ Ry @ Rz

    return rot_matrix

def augment_mesh(vertices, scale_min=0.9, scale_max=1.1, shift_strength=0.1, rotation_interval=90, rotation_axis=2, **kwargs):
    # random rotation around z axis
    Rotation_matrix = get_random_rotation_matrix_3d(interval_angle=rotation_interval, rotation_axis=rotation_axis)
    vertices = vertices @ Rotation_matrix
    
    # random scale
    scales = np.random.rand(3) * (scale_max - scale_min) + scale_min
    vertices = vertices * scales
    
    # random shift
    shift = (np.random.rand(3) * 2 - 1) * shift_strength
    vertices = vertices + shift
    
    # clip vertices
    vertices = np.clip(vertices, -0.999, 0.999)

    return vertices

    

def process_mesh_seam(vertices, faces, seam_edges, augment=False, augment_dict=None):
    """Process mesh vertices and faces."""

    # Transpose so that z-axis is vertical.
    
    vertices, center, v_scale = normalize_vertices(vertices, scale=0.85)

    vertices = vertices[:, [2, 0, 1]]

    if augment:
        vertices = augment_mesh(vertices, **augment_dict)

    mesh_data = {
        'vertices': vertices,
        'faces': faces,
        'lines': seam_edges,
    }    
    
    mesh_data = remove_duplicate_vertices_and_lines_for_seam(mesh_data)

    # Quantize and sort vertices, remove resulting duplicates, sort and reindex faces.
    mesh_data = sort_vertices_and_update_indices(mesh_data)

    mesh_data['v_scale'] = v_scale
    mesh_data['v_offset'] = center
    mesh_data['vertices'] = mesh_data['vertices'][:, [1, 2, 0]]

    # Discard degenerate meshes without faces.
    return mesh_data