import os
import numpy as np
import open3d as o3d
import matplotlib.cm as cm

from src.utils.np_utils import caculate_align_mat, undiscretize
from src.utils.base import split_graph_into_chains, split_and_filter_chains_1D

def glb2obj(glb_file, obj_file): 
    """
    Extract mesh with UV coordinates from a .glb file and save it as an .obj file.
    The .obj file will only contain vertices, UV coordinates, and faces; other data (e.g., normals, materials) will be ignored.

    Parameters:
    glb_file (str): Path to the input .glb file.
    obj_file (str): Path to the output .obj file for saving vertices, UVs, and faces.

    Returns:
    bool: True if the .glb file is loaded successfully with UV coordinates, False otherwise.
    """

    mesh = o3d.io.read_triangle_mesh(glb_file)

    if mesh.is_empty():
        return False
        
    if not mesh.has_triangle_uvs():
        return False
    
    vertices = np.asarray(mesh.vertices) # (num_vertices, 3)
    faces_v = np.asarray(mesh.triangles) # (num_faces, 3)
    uvs = np.asarray(mesh.triangle_uvs) # (num_faces*3, 2)

    faces_vt = np.arange(len(uvs)).reshape(-1, 3)

    unique_vertices, vertex_indices = np.unique(vertices, axis=0, return_inverse=True)
    unique_uvs, uv_indices = np.unique(uvs, axis=0, return_inverse=True)

    updated_faces_v = vertex_indices[faces_v]
    updated_faces_vt = uv_indices[faces_vt]

    faces = np.stack([updated_faces_v, updated_faces_vt], axis=1)

    faces += 1 # from 0-based to 1-based

    with open(obj_file, 'w') as file:
        for vertex in unique_vertices:
            file.write(f"v {' '.join(map(str, vertex))}\n")

        for uv in unique_uvs:
            file.write(f"vt {' '.join(map(str, uv))}\n")

        for face in faces:
            file.write(f"f {face[0][0]}/{face[1][0]} {face[0][1]}/{face[1][1]} {face[0][2]}/{face[1][2]} \n")
    
    return True

def save_chains_coords_to_seam_mesh(all_chains_coords, output_path, input_mesh=None, should_undiscretize=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mesh = o3d.geometry.TriangleMesh()
    num_chains = len(all_chains_coords)
    
    # 选择一个颜色图（例如viridis, inferno, etc.）
    colors = cm.rainbow(np.linspace(0, 1, num_chains))[..., :3]

    for i, chain_coords in enumerate(all_chains_coords):

        color = colors[i]
        
        if should_undiscretize:
            chain_coords = undiscretize(chain_coords)

        for idx in range(len(chain_coords) - 1):
            s_p = chain_coords[idx]
            e_p = chain_coords[idx + 1]

            mesh_cylinder = create_colored_cylinder(s_p, e_p, color=color)
            if mesh_cylinder is None:
                continue

            mesh += mesh_cylinder
    
    if input_mesh:
        input_mesh.paint_uniform_color([0.8, 0.8, 0.8])
        mesh += input_mesh

    o3d.io.write_triangle_mesh(output_path, mesh)
    
    

def generate_seam_mesh(vertices, faces, seam_edges, output_path):
    # mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color([0.8, 0.8, 0.8])

    for idx in range(len(seam_edges)):
        s_p = np.asarray(vertices[seam_edges[idx][0]])
        e_p = np.asarray(vertices[seam_edges[idx][1]])

        color = np.random.rand(3)
        
        mesh_cylinder = create_colored_cylinder(s_p, e_p, color=color)
        if mesh_cylinder is None:
            continue

        mesh += mesh_cylinder
    
    o3d.io.write_triangle_mesh(output_path, mesh)


def create_colored_cylinder(s_p, e_p, radius=0.01, color=[1, 0, 0]):
    vec_Arr = e_p - s_p
    vec_len = np.linalg.norm(vec_Arr)
    if vec_len <= 0.001: return None
    
    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, vec_len, 5, split=1)
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_cylinder.rotate(rot_mat, center=[0, 0, 0])
    mesh_cylinder.translate((s_p + e_p) / 2)
    
    mesh_cylinder.paint_uniform_color(color)
    
    return mesh_cylinder

def generate_chains_mesh(vertices, faces, seam_edges, chains_1D=None, output_path='./', color_mode='random'):
    # mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color([0.8, 0.8, 0.8])

    if chains_1D is None:
        chains = split_graph_into_chains(seam_edges)
    else:
        chains = split_and_filter_chains_1D(chains_1D)

    for i, chain in enumerate(chains):
        segments = np.vstack((chain[:-1], chain[1:])).T

        color = np.random.rand(3)

        for idx in range(len(segments)):
            s_p = np.asarray(vertices[segments[idx][0]])
            e_p = np.asarray(vertices[segments[idx][1]])

            mesh_cylinder = create_colored_cylinder(s_p, e_p, color=color)
            if mesh_cylinder is None:
                continue

            mesh += mesh_cylinder
    
    o3d.io.write_triangle_mesh(output_path, mesh)



def get_vertices_obb(vertices, noise_scale=1e-4):
    # 为了数值稳定性，添加一些噪声，避免所有点云都在一个平面上
    noise = noise_scale * np.random.uniform(-1, 1, vertices.shape)
    new_vertices = noise + vertices

    # 将 numpy 数组转换为 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(new_vertices)

    # 计算 OBB（有向包围盒）
    obb = point_cloud.get_oriented_bounding_box()

    return obb.center, obb.extent, obb.R