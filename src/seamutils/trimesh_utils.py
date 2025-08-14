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