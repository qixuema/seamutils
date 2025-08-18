import bpy

def build_mesh_from_data(unique_xyz, unique_uv, faces_xyz, faces_uv, name="TempMesh"):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # 创建几何
    mesh.from_pydata(unique_xyz, [], faces_xyz)
    mesh.update()

    # 建立 UV 层
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_data = uv_layer.data

    # 每个 loop 都要写 UV
    # faces_uv 与 faces_xyz 是对应的：逐面逐顶点
    loop_index = 0
    for poly, fuv in zip(mesh.polygons, faces_uv):
        for li, uvidx in zip(poly.loop_indices, fuv):
            uv_data[li].uv = unique_uv[uvidx]
            loop_index += 1

    return obj

def seams_from_uv(obj, clear_existing=True):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")
    if clear_existing:
        bpy.ops.uv.mark_seam(clear=True)
    bpy.ops.uv.seams_from_islands()
    bpy.ops.object.mode_set(mode='OBJECT')

def collect_xyz_faces_seams(obj):
    mesh = obj.data
    xyz = [tuple(v.co) for v in mesh.vertices]
    faces = [tuple(p.vertices) for p in mesh.polygons]
    seam_edges = [tuple(sorted(e.vertices)) for e in mesh.edges if e.use_seam]
    seam_edges = sorted(set(seam_edges))
    return xyz, faces, seam_edges

def extract_seam_bpy(unique_xyz, unique_uv, faces_xyz, faces_uv):
    # 1. 构建临时 Mesh
    obj = build_mesh_from_data(unique_xyz, unique_uv, faces_xyz, faces_uv)

    # 2. 从 UV 生成 seam
    seams_from_uv(obj, clear_existing=True)

    # 3. 收集结果
    xyz, faces_xyz_out, seam_edges = collect_xyz_faces_seams(obj)

    return xyz, faces_xyz_out, seam_edges
