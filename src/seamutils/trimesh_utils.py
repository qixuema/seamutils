import numpy as np
import trimesh
from qixuema.trimesh_utils import segments_to_prisms, polylines_to_mesh

def save_seam_mesh(output_path, xyz, faces, seam_edges, with_base_mesh=True, radius=0.1):
    if with_base_mesh:
        base_mesh = trimesh.Trimesh()
        base_mesh.vertices = xyz
        base_mesh.faces = faces
    else:
        base_mesh = None
        
    seam_segments = xyz[seam_edges]
    mesh = segments_to_prisms(
        seam_segments,
        base_mesh, 
        radius=radius,
    )
    
    mesh.export(output_path)
    

def chains_to_seam_mesh(vertices, faces, chains, with_base_mesh=False, radius=0.1):
    print(f'{len(chains)} chains')

    meshes = []
    
    if with_base_mesh:
        base_mesh = trimesh.Trimesh()
        base_mesh.vertices = vertices
        base_mesh.faces = faces
        base_mesh.visual.vertex_colors = np.tile([200,200,200,200], (base_mesh.vertices.shape[0], 1))

        meshes.append(base_mesh)
    
    seam_mesh = polylines_to_mesh(chains, radius=radius)
    meshes.append(seam_mesh)
    
    mesh = trimesh.util.concatenate(meshes)
    
    return mesh