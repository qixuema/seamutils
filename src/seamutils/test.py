import numpy as np

# 这里是你优化后的函数
def clean_invalid_faces(faces):
    diffs = np.abs(faces[:, [0, 0, 1]] - faces[:, [1, 2, 2]]).min(axis=1)
    mask = diffs < 0.5
    return faces[~mask]

def clean_invalid_lines(lines):
    diff = np.abs(lines[:, 0] - lines[:, 1])
    return lines[np.abs(diff) >= 0.5]
# 测试代码
def test_cleaning_functions():
    # 创建一个测试的 faces 数组，存储的是顶点索引值
    faces = np.array([
        [0, 1, 2],  # Valid face
        [0, 0, 0],  # Invalid face (same index for all vertices)
        [1, 2, 3],  # Valid face
        [4, 4, 4]   # Invalid face (same index for all vertices)
    ])
    
    # 创建一个测试的 lines 数组，存储的是顶点索引值
    lines = np.array([
        [0, 1],  # Invalid line (difference < 0.5)
        [1, 2],  # Valid line
        [2, 3],  # Invalid line (difference < 0.5)
        [3, 4]   # Valid line
    ])
    
    # 调用函数并输出结果
    cleaned_faces = clean_invalid_faces(faces)
    cleaned_lines = clean_invalid_lines(lines)

    print("Original Faces:\n", faces)
    print("Cleaned Faces:\n", cleaned_faces)
    print("\nOriginal Lines:\n", lines)
    print("Cleaned Lines:\n", cleaned_lines)

# 执行测试
test_cleaning_functions()
