import numpy as np

def read_obj_file(file_path):
    vertices = []
    faces = []
    lines = []
    vertice_normals = []

    with open(file_path, 'r') as file:
        for line in file:
            components = line.strip().split()

            if not components:
                continue

            if components[0] == 'v':
                # Parse vertex coordinates
                vertex = tuple(map(float, components[1:4]))
                vertices.append(vertex)
            elif components[0] == 'f':
                # Parse face indices (assuming triangular faces)
                face = tuple(map(int, components[1:4]))
                # face = [list(map(int, p.split('/'))) for p in components[1:]]
                faces.append(face)
                
            elif components[0] == 'l':
                # Parse face indices (assuming triangular faces)
                line = tuple(map(int, components[1:3]))
                lines.append(line)
            
            elif components[0] == 'vn':
                vertice_normal = tuple(map(float, components[1:4]))
                vertice_normals.append(vertice_normal)

    return {
        'vertices': np.array(vertices), 
        'faces': np.array(faces) - 1, 
        'lines': np.array(lines) - 1,
        'v_normals': np.array(vertice_normals)}

def write_obj_file(file_path, vertices, faces=None, vtx_colors=None, is_line=False, is_point=False):
    """
    Save a simple OBJ file with vertices and faces.

    Parameters:
    file_path (str): The path to save the OBJ file.
    vertices (list of tuples): A list of vertices, each vertex is a tuple (x, y, z).
    faces (list of tuples): A list of faces, each face is a tuple of vertex indices (1-based).
    """
    
    # 先转换为numpy数组
    vertices = np.array(vertices) if isinstance(vertices, list) else vertices
    faces = np.array(faces) if isinstance(faces, list) else faces
    
    if faces is not None:
        if faces.shape[1] == 2:
            is_line = True

    with open(file_path, 'w') as file:
        # Write vertices
        if vtx_colors is None:
            for vertex in vertices:
                    file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        else:
            for vertex, vtx_color in zip(vertices, vtx_colors):
                file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} {vtx_color[0]} {vtx_color[1]} {vtx_color[2]}\n")

        if is_point:
            return
        
        # Write faces
        if faces is None:
            return
        
        for face in faces:
            face_str = ' '.join([str(index + 1) for index in face])
            if is_line:
                file.write(f"l {face_str}\n")
            else:
                file.write(f"f {face_str}\n")
                

def save_obj_with_uv(output_path, unique_xyz, unique_uv, faces_xyz, faces_uv, float_fmt="{:.6f}"):
    """
    将 unique_xyz, unique_uv, faces_xyz, faces_uv 保存为带 UV 的三角面 OBJ 文件
    - 所有输入均为 numpy 数组
    - faces_xyz / faces_uv 形状为 (F, 3)，索引是 0-based
    """
    # 转为 numpy 数组以保证索引运算正常
    V  = np.asarray(unique_xyz, dtype=float)
    VT = np.asarray(unique_uv,  dtype=float)
    Fv = np.asarray(faces_xyz,  dtype=int)
    Fvt= np.asarray(faces_uv,   dtype=int)

    with open(output_path, "w", encoding="utf-8") as f:
        # 写顶点
        for x, y, z in V:
            f.write(f"v {float_fmt.format(x)} {float_fmt.format(y)} {float_fmt.format(z)}\n")
        # 写 UV（只取前两列）
        for u, v in VT[:, :2]:
            f.write(f"vt {float_fmt.format(u)} {float_fmt.format(v)}\n")
        # 写三角面（OBJ 索引是 1-based）
        for (v1, v2, v3), (t1, t2, t3) in zip(Fv, Fvt):
            f.write(f"f {v1+1}/{t1+1} {v2+1}/{t2+1} {v3+1}/{t3+1}\n")
