import cv2
import numpy as np
import os
from skimage.morphology import skeletonize
import itertools
from my_dlo_parser import CVFDLO_Parser



class Render_pipe:

    def __init__(self):
        self.cmap = self.voc_cmap(N=256, normalized=False)
        
    
    def voc_cmap(self, N=256, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        return cmap
    
    def render_instances(
        self,
        skel_merged,
        routes,
        ends_dict,
        end_pairs,
        dist_img,
        binary_mask
    ):

        H, W = binary_mask.shape
        canvas = np.zeros((H, W, 3), dtype=np.uint8)


        # =========================
        # Step2: 构建拓扑 instance
        # =========================
        parent = self.build_route_union_find(routes, ends_dict)
        instances = self.build_instances_from_union(parent)

        print(f"[Instance] 检测到 {len(instances)} 条线束")

        # =========================
        # Step3: 渲染每个 instance
        # =========================
        for inst_id, comp in enumerate(instances):

            # 🎨 颜色
            color_idx = ((inst_id + 1) * 37) % 256
            r, g, b = self.cmap[color_idx]
            color = (int(b), int(g), int(r))

            # -------------------------
            # A. 画 routes（主体）
            # -------------------------
            for r_id in comp:
                route_pixels = routes[r_id]['route']
                avg_radius = routes[r_id].get('width', (3,))[0]

                radius = max(int(avg_radius), 2)

                for y, x in route_pixels:
                    cv2.circle(canvas, (x, y), radius, color, -1)

            # -------------------------
            # B. 画 mergeEnds 连线（关键）
            # -------------------------
            for e_id, e_data in ends_dict.items():

                if e_data['route_label'] not in comp:
                    continue

                p1 = e_data['point']  # (x, y)

                for pe in e_data.get('pair_ends', []):
                    e2 = ends_dict[pe]

                    if e2['route_label'] not in comp:
                        continue

                    p2 = e2['point']

                    thickness = max(int(e_data.get('end_radius', 3) * 2), 2)

                    cv2.line(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)

        # =========================
        # Step4: Mask 裁剪
        # =========================
        canvas = cv2.bitwise_and(canvas, canvas, mask=binary_mask)

        return canvas


    def build_route_union_find(self, routes, ends_dict):
        parent = {r: r for r in routes.keys()}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pb] = pa

        for e_id, e_data in ends_dict.items():
            r1 = e_data['route_label']

            for pe in e_data.get('pair_ends', []):
                r2 = ends_dict[pe]['route_label']

                if r1 in parent and r2 in parent:
                    union(r1, r2)

        return parent
    
    def build_instances_from_union(self, parent):
        groups = {}

        def find_root(x):
            while parent[x] != x:
                x = parent[x]
            return x

        for r in parent:
            root = find_root(r)
            if root not in groups:
                groups[root] = []
            groups[root].append(r)

        return list(groups.values())
    
if __name__ == "__main__":
    # 请确保同级目录下有一张你的 HarnessHRNetV2 输出的二值化图片
    parser = CVFDLO_Parser(debug_dir="mydlo_debug", if_debug=True)
    out_dict = parser.run("../final_inference_mask.png")
    render = Render_pipe()
    
    canvas = render.render_instances( out_dict['skel_merged'],out_dict['routes'], out_dict['ends_dict'], 
                                     out_dict['end_pairs'],out_dict['dist_img'], out_dict['binary_mask'])
    
    cv2.imwrite("render.png", canvas)