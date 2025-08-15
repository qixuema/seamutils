import os
import numpy as np
import trimesh
from trimesh.creation import uv_sphere

from qixuema.np_utils import normalize_vertices
from seamutils.scipy_utils import build_vertex_adjacency_csr, one_ring_neighbors_csr, k_ring_csr

# ---------- 加载 OBJ ----------
def load_mesh_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    读取 .obj -> (vertices, faces[tri])
    自动处理 Scene、多子网格 & 三角化
    """
    mesh = trimesh.load(path, force='mesh', process=True)

    # 可能返回的是 Scene
    if isinstance(mesh, trimesh.Scene):
        # 将 Scene 中的几何合并
        geoms = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(geoms)

    # 确保是三角面
    if mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return vertices, faces


# ---------- 简单可视化导出（把一环染色保存为 PLY） ----------
def save_colored_ply(path, vertices, faces, highlight_ids=None, query_ids=None, base_col=(200,200,200), hi_col=(255,0,0)):
    V = vertices.shape[0]
    colors = np.tile(np.array(base_col, dtype=np.uint8), (V, 1))
    if highlight_ids is not None and len(highlight_ids) > 0:
        colors[np.asarray(highlight_ids, dtype=int)] = np.array(hi_col, dtype=np.uint8)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = np.hstack([colors, np.full((V,1), 255, dtype=np.uint8)])
    
    meshes = [mesh]
    if query_ids is not None:
        query_vertices = vertices[query_ids]
        query_mesh = uv_sphere(radius=0.02, count=[4, 4])
        query_mesh.vertices += query_vertices
        query_mesh.visual.face_colors = np.array([0, 255, 0, 1], dtype=np.uint8)
        meshes.append(query_mesh)
        
    if highlight_ids is not None:
        for highlight_id in highlight_ids:
            highlight_mesh = uv_sphere(radius=0.02, count=[4, 4])
            highlight_mesh.vertices += vertices[highlight_id]
            highlight_mesh.visual.face_colors = np.array([255, 0, 0, 1], dtype=np.uint8)
            meshes.append(highlight_mesh)
    
    meshes = trimesh.util.concatenate(meshes)
    
    meshes.export(path)

def test():
    obj_path = './test/data/6773dd0cdb09441d44eefb1e_Wolf3D_Outfit_Bottom_0_raw_mesh.obj'
    vertices, faces = load_mesh_obj(obj_path)
    vertices, center, scale = normalize_vertices(vertices)
    
    print(f"Loaded: V={vertices.shape[0]}, F={faces.shape[0]}")

    # 选几个点做测试（可改成你的索引列表）
    query = np.random.choice(vertices.shape[0], size=1, replace=False)
    print("Query indices:", query.tolist())

    # 1-ring
    neigh_dict, union_ids, union_pts = one_ring_neighbors_csr(vertices, faces, query)
    for q in query:
        print(f"v{q}: degree={len(neigh_dict[int(q)])}, neighbors={neigh_dict[int(q)][:10]}{'...' if len(neigh_dict[int(q)])>10 else ''}")
    print(f"Union 1-ring size: {len(union_ids)}")

    # 2-ring（可选）
    A = build_vertex_adjacency_csr(faces, vertices.shape[0])
    nbr2 = k_ring_csr(A, seeds=query, k=2, include_seeds=False)
    print(f"Union 2-ring size: {len(nbr2)}")

    # 导出一份把 1-ring 并集高亮的 PLY（方便用 Meshlab/CloudCompare 看）
    out_ply = "./test/data/one_ring_highlight.ply"
    save_colored_ply(out_ply, vertices, faces, highlight_ids=union_ids, query_ids=query)
    print(f"Saved colored PLY -> {out_ply}")

def main():
    test()

# ---------- 主流程 ----------
if __name__ == "__main__":
   main()