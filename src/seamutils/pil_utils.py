from PIL import Image, ImageDraw
import numpy as np
import time

def uv_fill(
    unique_uv, faces_uv, H=2048, W=2048,
    rgb=(0,0,255), bg=(255,255,255),
    edge_rgb=None, edge_px=1, supersample=2, wrap=False
):
    """
    生成类似“蓝色UV岛”的占据图 (轻依赖: Pillow)
    """
    uv = np.asarray(unique_uv, np.float32)
    F  = np.asarray(faces_uv,   np.int32)

    ss = max(1, int(supersample))
    Hs, Ws = H*ss, W*ss

    if wrap:
        uv = np.mod(uv, 1.0)

    # UV -> 像素坐标（翻转V轴）
    x = np.clip(uv[:,0], 0, 1) * (Ws - 1)
    y = (1.0 - np.clip(uv[:,1], 0, 1)) * (Hs - 1)
    pts = np.stack([x, y], axis=1)

    img = Image.new('RGB', (Ws, Hs), color=bg)
    drw = ImageDraw.Draw(img)

    # 逐面绘制（三角形很快；Pillow 只有边界像素级别的AA，靠超采样提升质量）
    for (i0,i1,i2) in F:
        tri = [(pts[i0,0], pts[i0,1]),
               (pts[i1,0], pts[i1,1]),
               (pts[i2,0], pts[i2,1])]
        drw.polygon(tri, fill=rgb, outline=edge_rgb if edge_rgb else None)

        if edge_rgb and edge_px>0:
            # Pillow 线宽不影响 polygon 的 outline 填充，这里再画一遍线条
            drw.line(tri + [tri[0]], fill=edge_rgb, width=max(1, edge_px*ss))

    # 下采样做抗锯齿
    if ss > 1:
        img = img.resize((W, H), resample=Image.LANCZOS)

    return img


def test():
    npz_path = '/root/studio/datasets/seam/part/677514ad6b329cf708d74917.glb_nogroup_v3/677514ad6b329cf708d74917_Wolf3D_Outfit_Bottom_0.npz'

    sample = np.load(npz_path)
    vertices = sample['vertices']
    faces = sample['faces']
    uv = sample['uv']
    faces_uv = sample['faces_uv']

    num_trials = 10

    time_list = []

    resolution = 512
    supersample = 2

    print(f'num_uv: {uv.shape}')
    print(f'num_faces_uv: {faces_uv.shape}')
    print(f'resolution: {resolution}')
    print(f'supersample: {supersample}')

    for i in range(num_trials):
        # 用法：
        start_time = time.time()
        img = uv_fill(uv, faces_uv, 512, 512, rgb=(0,0,255), edge_rgb=(255,255,255), supersample=2)
        end_time = time.time()
        time_list.append(end_time - start_time)

    print(f'Average time: {np.mean(time_list)} seconds')

    # img.save('uv_fill.png')
    

if __name__ == '__main__':
    test()