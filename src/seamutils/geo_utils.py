import numpy as np
from qixuema.np_utils import deduplicate_lines, deduplicate_faces

def create_polygon_pillar_from_line(line, n_sides=6, radius=0.1):
    """
    根据给定的线段生成N边形柱体, 底面为N边形。
    
    参数：
        line (tuple): 包含两个端点的线段，格式为((x1, y1, z1), (x2, y2, z2))
        n_sides (int): N边形的边数
        width (float): 即底面N边形的半径
    
    返回：
        vertices (ndarray): 顶点坐标数组
        faces (list): 顶点索引构成的面列表
        colors (ndarray): 随机颜色数组，用于指定每个柱体的颜色
    """
    point1, point2 = line
    # 计算线段的方向向量
    direction = np.array(point2) - np.array(point1)
    # 计算法向量，简单地取垂直于方向的向量
    normal = np.cross(direction, [0, 0, 1])
    if np.linalg.norm(normal) == 0:
        normal = np.cross(direction, [1, 0, 0])
    
    normal = normal / np.linalg.norm(normal)  # 归一化
    
    # 计算N边形的顶点
    angle = 2 * np.pi / n_sides  # 每个顶点之间的角度
    polygon_points = []
    
    for i in range(n_sides):
        angle_offset = i * angle
        offset = np.array([np.cos(angle_offset), np.sin(angle_offset), 0]) * radius
        polygon_points.append(point1 + offset)
    
    # 生成柱体的上面顶点，使用相同的旋转方向，将其偏移到线段的另一个端点
    top_polygon_points = []
    for point in polygon_points:
        top_polygon_points.append(point + direction)
    
    # 合并底面和顶面顶点
    vertices = np.array(polygon_points + top_polygon_points)
    
    # 创建面（底面和顶面相连的侧面）
    faces = []
    for i in range(n_sides):
        next_i = (i + 1) % n_sides
        faces.append([i, next_i, n_sides + next_i, n_sides + i])  # 一个四面体的连接

    # 为每个顶点设置颜色（红色）
    vertex_colors = np.array([[1.0, 0.0, 0.0]] * len(vertices))  # 所有顶点为红色
    
    return vertices, faces, vertex_colors

def generate_polygon_pillars(lines, n_sides=6, radius=0.1):
    all_vertices = []
    all_faces = []
    all_colors = []
    
    vertex_offset = 0  # 用于处理每个柱体的顶点索引，避免重复
    
    for line in lines:
        vertices, faces, vertex_colors = create_polygon_pillar_from_line(line, n_sides, radius)
        
        # 处理面索引，确保全局唯一
        faces = np.array(faces) + vertex_offset
        
        # 合并顶点、面和颜色
        all_vertices.append(vertices)
        all_faces.append(faces)
        all_colors.append(vertex_colors)
        
        # 更新顶点偏移量
        vertex_offset += len(vertices)
    
    # 合并所有数据
    all_vertices = np.vstack(all_vertices)
    all_faces = np.vstack(all_faces)
    all_colors = np.vstack(all_colors)
    
    return {
        'vertices': all_vertices,
        'faces': all_faces,
        'vtx_colors': all_colors
    }



def remove_duplicate_vertices_and_lines(mesh:dict, return_indices=False, tolerance=0.0001, return_rows_changed=False):
    # 注意，在这部分的代码中，我们并没有对顶点的顺序进行排序，我们只是剔除了重复（三维空间接近）的顶点,
    vertices, lines = mesh['vertices'], mesh['lines']

    # Example tolerance value
    # Adjust this as per your requirement

    # Adjust points based on tolerance
    adjusted_points = np.round(vertices / tolerance) * tolerance
    
    # 移除重复的点并创建索引映射    
    unique_points, inverse_indices = np.unique(adjusted_points, axis=0, return_inverse=True)

    updated_lines = inverse_indices[lines] # 更新线段索引，但是线段的顺序还是不变
    
    # 删除重复的线段，确保每条线段的小索引在前，大索引在后
    # 保存排序前的数组以便后续比较
    updated_lines_before = np.copy(updated_lines)

    updated_lines = np.sort(updated_lines, axis=1)
    # 比较排序前后的行是否发生了变化，生成一个布尔数组，表示每一行是否改变
    changed_rows = np.any(updated_lines_before != updated_lines, axis=1)
    
    unique_lines, indices = np.unique(updated_lines, axis=0, return_index=True) # 这里 unique 之后，lines 的顺序会被打乱
    
    sorted_indices = np.argsort(indices)
    unique_lines = unique_lines[sorted_indices] # 因此这里对 line 的顺序进行了重新排序，恢复原有的顺序，这是有必要的


    # If faces are provided, update the faces' indices
    if 'faces' in mesh:
        faces = mesh['faces']
        updated_faces = inverse_indices[faces]
        
        # Sort faces for uniqueness, similar to lines
        updated_faces_before = np.copy(updated_faces)
        updated_faces = np.sort(updated_faces, axis=1)
        changed_faces = np.any(updated_faces_before != updated_faces, axis=1)
        
        unique_faces, face_indices = np.unique(updated_faces, axis=0, return_index=True)
        sorted_face_indices = np.argsort(face_indices)
        unique_faces = unique_faces[sorted_face_indices]
        
        unique_faces = clean_invalid_faces(unique_faces)
        
        # Update faces in the lineset
        mesh['faces'] = unique_faces


    mesh['vertices'] = unique_points
    
    valid_lines = clean_invalid_lines(unique_lines)
    
    mesh['lines'] = valid_lines
    
    # if return_rows_changed:
    #     return mesh, indices, sorted_indices, changed_rows
    # elif return_indices:
    #     return mesh, indices, sorted_indices
    
    return mesh

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

if __name__ == "__main__":

    # 示例：创建一个包含多个线段的lineset
    lines = [
        ((0, 0, 0), (1, 0, 0)),
        ((1, 0, 0), (1, 1, 0)),
        ((1, 1, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 0, 0))
    ]

    # 生成带颜色的N边形柱体（不进行绘制）
    pillars = generate_polygon_pillars(lines, n_sides=6, radius=0.1)

    # 输出每个柱体的顶点、面和颜色
    for i, (vertices, faces, color) in enumerate(pillars):
        print(f"柱体 {i + 1}:")
        print("顶点:")
        print(vertices)
        print("面:")
        print(faces)
        print("颜色:")
        print(color)
        print()