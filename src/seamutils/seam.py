import numpy as np
import os
from pathlib import Path

from qixuema.np_utils import normalize_vertices, deduplicate_lines, deduplicate_faces

from seamutils.np_utils import filter_edges
from seamutils.graph_utils import build_graph
from seamutils.geo_utils import faces_to_edges


def xyz_indices_to_uv_indices(xyz_indices, faces_xyz, faces_uv):
    """
    xyz_indices : list[int]    # 在 unique_xyz 中的索引
    faces_xyz : (F,3)  # 每个面的几何顶点索引
    faces_uv  : (F,3)  # 每个面的 UV 顶点索引
    返回: np.ndarray (n_uv,)  # 对应的 unique_uv 索引集合
    """
    # mask: 哪些面顶点等于这个几何点
    vertice_mask = np.isin(faces_xyz, xyz_indices)
    
    face_mask = np.any(vertice_mask, axis=1)
    
    # 用 mask 在 faces_uv 里取出对应的 uv 索引
    uv_indices = faces_uv[vertice_mask]
    # 去重
    return np.unique(uv_indices), face_mask



def extract_seams(xyz, uv, faces_xyz, faces_uv, edges_xyz=None, tolerance=1e-6, tgt_dir=None, file_name=None):
    
    if edges_xyz is None:
        edges_xyz = faces_to_edges(faces_xyz)
    
    # xyz process

    # xyz, _, _ = normalize_vertices(xyz, scale=1.0)
    # uv, _, _ = normalize_vertices(uv, scale=1.0)
    
    rounded_xyz = np.round(xyz / tolerance) * tolerance
    
    uv_tolerance = min(tolerance, 1e-6)
    rounded_uv = np.round(uv / uv_tolerance) * uv_tolerance
    
    xyz_unique, inverse = np.unique(rounded_xyz, axis=0, return_inverse=True)
    
    # update faces_xyz and edges_xyz with deduplicated xyz

    faces_xyz_updated = inverse[faces_xyz]
    edges_xyz_updated = inverse[edges_xyz]


    # edges deduplication
    sorted_edges_xyz_updated = np.sort(edges_xyz_updated, axis=1) # 需要注意一下，这里对线段内的方向进行了更新
    
    edges_xyz_unique, indices = np.unique(sorted_edges_xyz_updated, axis=0, return_index=True)
    sorted_indices = np.argsort(indices)
    edges_xyz_unique = edges_xyz_unique[sorted_indices]

    # uv process
    
    uv_unique, inverse = np.unique(rounded_uv, axis=0, return_inverse=True)

    # update faces_uv and edges_uv with deduplicated uv

    faces_uv_updated = inverse[faces_uv]

    # faces deduplication
    faces_xyz_updated, faces_uv_updated = deduplicate_faces(faces_xyz_updated, faces_uv_updated)

    # split mesh by xyz
    subgraphs = build_graph(edges_xyz_unique)
    
    parts = []
    
    for part_id, sg in enumerate(subgraphs):
        nodes = list(sg.nodes())
        nodes.sort()
    
        uv_indices, face_mask = xyz_indices_to_uv_indices(nodes, faces_xyz_updated, faces_uv_updated)
        sub_xyz = xyz_unique[nodes]
        sub_faces_xyz = faces_xyz_updated[face_mask]
        sub_uv = uv_unique[uv_indices]
        sub_faces_uv = faces_uv_updated[face_mask]
        
        xyz_map = np.full(xyz_unique.shape[0], -1, dtype=np.int32)
        xyz_map[nodes] = np.arange(len(nodes), dtype=np.int32)
        
        uv_map = np.full(uv_unique.shape[0], -1, dtype=np.int32)
        uv_map[uv_indices] = np.arange(len(uv_indices), dtype=np.int32)
        
        sub_faces_xyz = xyz_map[sub_faces_xyz]
        sub_faces_uv = uv_map[sub_faces_uv]

        # output_path = f'_{comp_id}.obj'
        # save_obj_with_uv(output_path, sub_xyz, sub_uv, sub_faces_xyz, sub_faces_uv)
        
        assert sub_faces_xyz.shape[0] == sub_faces_uv.shape[0]
        
        # faces = np.concatenate([sub_faces_xyz, sub_faces_uv], axis=-1)
        # # remove zero area faces
        # zero_uv_area_faces_mask = find_zero_area_faces_mask(sub_faces_uv, sub_uv)
        # valid_faces = faces[~zero_uv_area_faces_mask]
        # seam_edges = extract_seams_by_faces(valid_faces)
        
        
        new_file_name = f'{file_name}_{part_id}.npz'
        new_npz_file_path = os.path.join(tgt_dir, new_file_name)
        if Path(new_npz_file_path).exists():
            continue
        
        from seamutils.bpy_utils import extract_seam_bpy
        
        sub_xyz, sub_faces_xyz, seam_edges = extract_seam_bpy(sub_xyz, sub_uv, sub_faces_xyz, sub_faces_uv)
        
        # 检查 sub_xyz 里面没有 nan 或者 inf
        if np.isnan(sub_xyz).any() or np.isinf(sub_xyz).any():
            print(f'sub_xyz has nan or inf, skip part {part_id} of {file_name}')
            continue
        
        sub_xyz = np.array(sub_xyz)
        sub_faces_xyz = np.array(sub_faces_xyz)
        seam_edges = np.array(seam_edges)

        assert sub_faces_uv.shape[0] == sub_faces_xyz.shape[0]
            
        # filter seam edges
        if len(seam_edges) == 0:
            # print(f'no seam edges for {part_id}')
            continue
    
        seam_edges = deduplicate_lines(seam_edges)
    
        seam_edges, valid_mask = filter_edges(seam_edges)
        
        new_sample = {
            'xyz': sub_xyz,
            'faces_xyz': sub_faces_xyz,
            'seam_edges': seam_edges,
            'uv': sub_uv,
            'faces_uv': sub_faces_uv,
        }
        
        parts.append(new_sample)
        
    return parts    
    

def extract_seams_from_mesh(mesh, tolerance=1e-6):
    """
        mesh trimesh.Trimesh object
        tolerance float, default 1e-3
    Returns:
        xyz_unique: np.ndarray, shape (N, 3)
        faces_xyz_updated: np.ndarray, shape (M, 3)
        seam_edges: np.ndarray, shape (K, 2)
    """

    xyz = mesh.vertices
    faces_xyz = mesh.faces
    faces_uv = mesh.faces
    
    uv = mesh.visual.uv

    edges_xyz = mesh.edges
    
    return extract_seams(xyz, uv, faces_xyz, faces_uv, edges_xyz, tolerance)