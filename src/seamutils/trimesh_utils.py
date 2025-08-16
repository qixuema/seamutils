import numpy as np
import trimesh
from qixuema.trimesh_utils import segments_to_prisms

def save_seam_mesh(output_path, xyz, faces, seam_edges):
    base_mesh = trimesh.Trimesh()
    base_mesh.vertices = xyz
    base_mesh.faces = faces
    
    seam_segments = xyz[seam_edges]
    mesh = segments_to_prisms(
        seam_segments,
        base_mesh, 
    )
    
    mesh.export(output_path)
    

def chains_to_seam_mesh(vertices, faces, chains, output_path='./'):
    base_mesh = trimesh.Trimesh()
    base_mesh.vertices = vertices
    base_mesh.faces = faces

    print(f'{len(chains)} chains')

    meshes = []
    meshes.append(base_mesh)
    
    for i, chain in enumerate(chains):
        segments = np.stack([chain[:-1], chain[1:]], axis=1)
        color = np.random.rand(3)

        chain_mesh = segments_to_prisms(
            segments,
            color=color,
        )

        meshes.append(chain_mesh)

    mesh = trimesh.util.concatenate(meshes)
    
    return mesh

    
    