import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

# -------- 构建顶点邻接矩阵（CSR） --------
def build_vertex_adjacency_csr(faces: np.ndarray, n_vertices: int | None = None) -> csr_matrix:
    """
    从三角面 (F,3) 构建无向顶点邻接矩阵 A(形状: V x V)，A[i,j]=1 表示顶点 i 与 j 相邻。
    去自环、去重。返回 CSR 稀疏矩阵。
    """
    faces = np.asarray(faces, dtype=np.int64)
    if n_vertices is None:
        n_vertices = int(faces.max()) + 1

    # 三角面 -> 边
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])                 # (3F, 2)

    # 无向图：加上反向边
    ij = np.vstack([edges, edges[:, ::-1]])            # (6F, 2)
    data = np.ones(len(ij), dtype=np.uint8)

    A = coo_matrix((data, (ij[:, 0], ij[:, 1])), shape=(n_vertices, n_vertices))
    A.setdiag(0)                                       # 去自环
    A.eliminate_zeros()
    return A.tocsr()

# -------- 1-ring 批量查询 --------
def one_ring_neighbors_csr(vertices: np.ndarray,
                           faces: np.ndarray,
                           query_idx,
                           return_union: bool = False,
                           return_coords: bool = False):
    """
    批量返回 query_idx 的 1-ring 邻域。
    Returns:
        neigh_dict: {vid -> np.ndarray(邻居顶点索引)}
        union_ids:  所有查询点的邻域并集 (np.ndarray)
        union_pts:  对应坐标 (K,3) 若 return_coords=True
    """
    Vn = vertices.shape[0]
    A = build_vertex_adjacency_csr(faces, Vn)

    q = np.atleast_1d(np.asarray(query_idx, dtype=np.int64))
    # 单点邻居：直接查稀疏行的列索引
    neigh_dict = {int(i): A[int(i)].indices for i in q}

    if return_union:
        # 多点并集：用布尔掩码一次性取多行并集
        mask = np.zeros(Vn, dtype=bool); mask[q] = True
        # 选出这些行相加，>0 的列即为邻居并集
        union_mask = (A[mask].sum(axis=0) > 0).A1
        # 去掉查询点自身
        union_mask[q] = False
        union_ids = np.flatnonzero(union_mask)
        
        return neigh_dict, union_ids

    if return_coords:
        coords_dict = {i: vertices[idxs] for i, idxs in neigh_dict.items()}
        return neigh_dict, coords_dict

    return neigh_dict

# -------- 可选：k-ring（2环/3环…）批量查询 --------
def k_ring_csr(A: csr_matrix, seeds, k: int, include_seeds: bool = False) -> np.ndarray:
    """
    在邻接矩阵 A 上做 BFS，返回 seeds 的 k-ring（并集）。
    A: 顶点-顶点邻接 CSR
    seeds: int 或 1D array
    """
    Vn = A.shape[0]
    seeds = np.atleast_1d(np.asarray(seeds, dtype=np.int64))
    visited = np.zeros(Vn, dtype=bool)
    frontier = np.zeros(Vn, dtype=bool)
    visited[seeds] = True
    frontier[seeds] = True

    for _ in range(k):
        rows = np.flatnonzero(frontier)
        if rows.size == 0:
            break
        # 所有 frontiers 的邻居并集：取这些行相加
        nxt = (A[rows].sum(axis=0) > 0).A1
        nxt &= ~visited
        visited |= nxt
        frontier = nxt

    out = visited if include_seeds else (visited & ~np.isin(np.arange(Vn), seeds))
    return np.flatnonzero(out)

# ============================================================
#  一次构建邻接矩阵，然后 for 循环逐点（自回归）查邻域
# ============================================================

class VertexNeighborhood:
    """
    先构建一次 CSR 邻接；随后可：
      - get_neighbors(i): 取顶点 i 的 1-ring 邻居
    """
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.asarray(vertices)
        self.A = build_vertex_adjacency_csr(faces, n_vertices=len(vertices))

    def get_neighbors(self, i: int, return_coords: bool = False) -> np.ndarray:
        i = int(i)
        neigh = self.A[i].indices
        if return_coords:
            return neigh, self.vertices[neigh]
        return neigh

def count_uv_islands(faces_uv: np.ndarray, return_labels: bool = False):
    """
    按“共享任意 UV 顶点”连通来统计 UV 岛。
    """
    faces_uv = np.asarray(faces_uv, dtype=np.int64)
    F = faces_uv.shape[0]
    if F == 0:
        return (0, np.empty((0,), dtype=np.int64)) if return_labels else 0

    flat = faces_uv.ravel()
    face_ids = np.repeat(np.arange(F, dtype=np.int64), faces_uv.shape[1])

    # 过滤无效 UV 索引
    valid = flat >= 0
    flat = flat[valid]
    face_ids = face_ids[valid]

    if flat.size == 0:
        # 没有任何有效 UV：每张面自成一个岛
        labels = np.arange(F, dtype=np.int64)
        return (F, labels) if return_labels else F

    U = int(flat.max()) + 1  # UV 顶点数（索引最大值 + 1）

    # 面-UV 顶点的稀疏关联矩阵 B (F x U)
    data = np.ones_like(face_ids, dtype=np.bool_)
    B = coo_matrix((data, (face_ids, flat)), shape=(F, U), dtype=bool).tocsr()

    # 面-面邻接：是否共享任意一个 UV 顶点
    # 注意：布尔乘法更省内存；结果包含对角线（自连接），正好保留孤立面
    A = (B @ B.T).astype(bool).tocsr()

    # 连通分量 = UV 岛
    n_islands, labels = connected_components(A, directed=False, return_labels=True)

    if return_labels:
        return int(n_islands), labels.astype(np.int64, copy=False)
    else:
        return int(n_islands)
