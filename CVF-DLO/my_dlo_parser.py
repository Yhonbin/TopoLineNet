import cv2
import numpy as np
import os
from skimage.morphology import skeletonize
import itertools

class CVFDLO_Parser:
    def __init__(self, debug_dir="./mydlo_debug", if_debug=False):
        self.debug_dir = debug_dir
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)      
        self.if_debug = if_debug
        self.kernel_size = 4
        self.cmap = self.voc_cmap(N=256, normalized=False)
        self.total_mean_width = None # 线束平均宽度
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
        
    def run(self, mask_path):
        """
        核心运行函数：极其干净的流程式调用，仅包含逻辑模块与可视化模块的组合。
        """
        print(f"--- 启动 My-DLO 拓扑解析器: {mask_path} ---")
        
        # 0. 读取输入
        binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        H,W = binary_mask.shape
        self.kernel_size = max(H,W) // 200
        # 1. 骨架化
        skeleton = skeletonize(binary_mask)
        
        if self.if_debug: 
            skeleton_uint8 = np.uint8(skeleton) * 255
            vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            vis[skeleton_uint8 == 255] = [0, 0, 255] 
            cv2.imwrite(os.path.join(self.debug_dir, "01_skeleton.png"), vis)   
            
        # 2. 提取交叉点
        intersections = self.extractInts(skeleton)
        
        if self.if_debug:
            self.vis_topology(skeleton_uint8, None, intersections.T)
            
        # 提前计算距离变换图
        dist_img = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
        
        # 3. 自适应聚类交叉点，生成中心交叉点
        ints_dict, merged_centers, _ = self.adaptive_cluster_intersections(binary_mask, intersections.T, dist_img,overlap_factor=1.2)

        if self.if_debug:
            self.vis_clustered_centers(skeleton_uint8, merged_centers,ints_dict) # 可视化聚类结果
            
        # 4. 挖洞
        skel_broken, radii_info = self.distance_transform_and_break(skeleton_uint8, ints_dict)
        
        if self.if_debug:
            self.vis_erasure(skeleton_uint8, skel_broken, dist_img, radii_info) # 可视化挖洞结果
        
        # 5. label关联中心交叉点
        num_labels, labels, ints_dict,window_size = self.label_routes_and_associate(skel_broken, ints_dict, threshold=self.kernel_size)
        if self.if_debug:
           self.vis_route_labels(labels, ints_dict, window_size) # 可视化最终关联结果
        
        labels, skel_broken, ints_dict = self.prune_labels_with_intersections(labels,skel_broken, ints_dict)
        # ⭐ 可选：重新编号（推荐）
        num_labels, labels = cv2.connectedComponents((labels > 0).astype(np.uint8))
        
        # 6. 从断开骨架中提取路径，并记录每条路径的宽度信息
        ends = self.extractEndslist(skel_broken)
        routes, labels, ends = self.extractRoutes(ends, num_labels, labels, skel_broken, ints_dict)
        if self.if_debug:
           self.showRoutes(routes, skel_broken, connect=False, mask=binary_mask) # 可视化最终路径结果
           
        # 记录route中每一个点的宽度信息（通过距离场估计）
        routes = self.estimateRoutewidthFromSegment(routes, dist_img)
        
        # 7. 关联端点与交叉点、路径
        ends_dict, ints_dict = self.constructEndsDict_New(routes, labels, ends, ints_dict)
        
        # 8. 最后一步：基于端点与路径的关联关系，进行最终的认亲匹配，输出完整的端点字典
        end_pairs = self.execute_local_combinatorial_matching(ints_dict, ends_dict, routes)
        
        # 注入端点配对关系
        ends_dict = self.inject_pairs_into_ends_dict(ends_dict, end_pairs)

        skel_reconstructed = self.mergeEnds(skel_broken, ends_dict, end_pairs)
        
        if self.if_debug:
           self.showRoutes(routes, skel_reconstructed,connect=True, mask=binary_mask) # 可视化最终路径结果
        
        return {
                "skel_merged": skel_reconstructed,   # ⭐ mergeEnds后的骨架（最关键）
                "routes": routes,                   # 用来提供 seed
                "end_pairs": end_pairs,             # 端点对
                "ends_dict": ends_dict,             # 可选（用于辅助）
                "dist_img": dist_img,               # 用来恢复宽度
                "binary_mask": binary_mask          # 最后裁剪边缘
            }
        
        
        
    def inject_pairs_into_ends_dict(self, ends_dict, end_pairs):
        for e1, e2 in end_pairs:
            if 'pair_ends' not in ends_dict[e1]:
                ends_dict[e1]['pair_ends'] = []
            if 'pair_ends' not in ends_dict[e2]:
                ends_dict[e2]['pair_ends'] = []

            ends_dict[e1]['pair_ends'].append(e2)
            ends_dict[e2]['pair_ends'].append(e1)

        return ends_dict
    def prune_labels_with_intersections(self, labels,skel, ints_dict):
        """
        在 extractRoutes 之前，对 labels 进行剪枝（去掉 H 中间短路径）
        """

        new_labels = labels.copy()
        new_skel = skel.copy()
        unique_labels = [l for l in np.unique(labels) if l != 0]

        # =========================
        # Step1: 构建 label → intersections
        # =========================
        label_to_ints = {l: [] for l in unique_labels}

        for int_id, int_dict in ints_dict.items():
            for l in int_dict["routes_label"]:
                if l in label_to_ints:
                    label_to_ints[l].append(int_id)

        # =========================
        # Step2: 遍历每个连通域
        # =========================
        for l in unique_labels:
            pixels = np.argwhere(labels == l)
     
            length = len(pixels)
            int_ids = label_to_ints.get(l, [])
            # =========================
            # ⭐ H结构判定
            # =========================
            cond1 = length < self.kernel_size * 6
            cond2 = len(int_ids) == 2
            if cond1 and cond2:
                # 删除该路径
                new_labels[labels == l] = 0
                new_skel[labels == l] = 0
                # =========================
                # ⭐ 合并交叉点
                # =========================
                int_id1, int_id2 = int_ids

                if int_id1 in ints_dict and int_id2 in ints_dict:

                    p1 = ints_dict[int_id1]
                    p2 = ints_dict[int_id2]

                    y1, x1 = p1["point"]
                    y2, x2 = p2["point"]

                    cy = int((y1 + y2) / 2)
                    cx = int((x1 + x2) / 2)

                    d = self.distance2D((y1, x1), (y2, x2))
                    new_r = max(p1["int_radius"], p2["int_radius"]) + d / 2

                    # 合并
                    ints_dict[int_id1]["point"] = (cy, cx)
                    ints_dict[int_id1]["int_radius"] = new_r

                    # 删除第二个
                    del ints_dict[int_id2]
                    
        for int_dict in ints_dict.values():
            int_dict["routes_label"] = [
                l for l in int_dict["routes_label"] if l in np.unique(new_labels)
            ]
        return new_labels, new_skel, ints_dict
    
    

    def vis_route_labels(self, labels_im, ints_dict,window_size):
        """可视化：用随机颜色渲染不同的线束段，并圈出交叉点的搜索窗口"""
        H, W = labels_im.shape
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 为每条线段生成随机颜色 (为了明显，避免纯黑)
        np.random.seed(42) # 固定种子，保证每次颜色一样
        num_labels = np.max(labels_im) + 1
        colors = np.random.randint(50, 255, size=(num_labels, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0] # 背景保持黑色
        
        # 上色
        vis = colors[labels_im]
        
        # 绘制交叉点关联范围
        for k, int_dict in ints_dict.items():
            cy, cx = int_dict['point']
            # 画一个白色的虚线/细线方框表示认亲窗口
            cv2.rectangle(vis, 
                          (max(0, cx - window_size), max(0, cy - window_size)), 
                          (min(W, cx + window_size), min(H, cy + window_size)), 
                          (255, 255, 255), 1)
            # 标出 ID
            cv2.putText(vis, f"Int {k}", (cx - 10, cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(os.path.join(self.debug_dir, "06_associated_routes.png"), vis)

    def label_routes_and_associate(self, skel_broken, ints_dict, threshold):
        """
        步骤 5：连通域标记与交叉点拓扑关联
        :param skel_broken: Step 4 产生的断开骨架
        :param ints_dict: Step 3 产生的交叉点字典
        :param threshold: 即 kernel_size，Step 3 产生的聚类阈值
        :return: labels_im (带标签的图像), ints_dict (更新了 routes_label 的字典)
        """
        print("\n--- Running Step 5: Topology Graph Association ---")
        
        # 1. 连通域分析 (给每一段断开的线束打上唯一的整数 ID)
        # num_labels 是总连通域数量，labels_im 是和原图一样大的矩阵，背景为 0，线段为 1,2,3...
        num_labels, labels_im = cv2.connectedComponents(skel_broken)
        print(f"提取了 {num_labels - 1} 条独立的线束路径段 (Routes)。")

        H, W = labels_im.shape
        # 2. 交叉点与路径的匹配 (改进的安全版源码)
        for k, int_dict in ints_dict.items():
            point = int_dict['point']
            radius = int_dict['int_radius']
            # 搜索窗口大小：黑洞半径 + 聚类阈值 (kernel_size)
            window_size = int(round(radius) + threshold)
            # 安全计算边界框 (防越界)
            y_min = max(0, point[0] - window_size)
            y_max = min(H, point[0] + window_size)
            x_min = max(0, point[1] - window_size)
            x_max = min(W, point[1] + window_size)
            
            # 提取局部窗口
            label_cover = labels_im[y_min:y_max, x_min:x_max]
            
            # 获取该窗口内所有非 0 的唯一标签 ID
            associated_routes = [int(v) for v in np.unique(label_cover) if v != 0]
            int_dict['routes_label'] = associated_routes
            
            print(f"交叉点 {k} (坐标 {point}) 关联了线段 ID: {associated_routes}")

        return num_labels, labels_im, ints_dict, window_size
        

    def distance_transform_and_break(self, skeleton_img, ints_dict):
        """
        步骤 4：直接利用 ints_dict 中保存的 int_radius 进行黑洞吞噬
        """
        skel_broken = skeleton_img.copy()
        radii_info = [] 
        
        for idx, int_dict in ints_dict.items():
            cy, cx = int_dict['point']
            # 提取源码计算出的动态膨胀半径
            # 乘以 1.2 依然是一个良好的工程习惯，确保断开干净
            erase_radius = int(int_dict['int_radius']) 
            
            radii_info.append(((cx, cy), erase_radius))
            
            # 挖去交叉点黑洞
            cv2.circle(skel_broken, (cx, cy), erase_radius, 0, -1)
            
        return skel_broken, radii_info
    
    
    def showRoutes(self, routes, skel, connect=False, mask=None):
        if mask is not None:
            back_white = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            back = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
            back_white = np.ones_like(back) * 255
        for i in routes.keys():
            for point in routes[i]['route']:
                back_white[point[0]][point[1]] = self.cmap[i]
        if connect:
            # todo
            cv2.imwrite(os.path.join(self.debug_dir, "08_show_routes_connect.png"), skel)

        else:
            cv2.imwrite(os.path.join(self.debug_dir, "07_show_routes.png"), back_white)
    
        
    def vis_clustered_centers(self, skeleton_img, merged_centers, ints_dict):
        """可视化：展示聚类后的真实几何中心点"""
        vis = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
        vis[skeleton_img == 255] = [200, 200, 200]

        for idx, int_dict in ints_dict.items():
            cy, cx = int_dict['point']
            radius = int(int_dict['int_radius'])
            # 画出聚类的影响范围
            cv2.circle(vis, (cx, cy), radius, (255, 200, 200), 1)
            # 画出最终中心点
            cv2.circle(vis, (cx, cy), 3, (0, 165, 255), -1) # 橙色：真正的路口中心
            
        cv2.imwrite(os.path.join(self.debug_dir, "03_clustered_centers.png"), vis)
    
    
    def vis_erasure(self, skeleton_img, skel_broken, edt_map, radii_info):
        """可视化：距离场黑白图，以及最终的黑洞打断效果"""
        # 保存归一化的距离场图
        edt_vis = cv2.normalize(edt_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(self.debug_dir, "04_distance_transform.png"), edt_vis)
        
        # 保存打断效果图
        vis = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
        vis[skel_broken == 255] = [200, 200, 200] # 残留的路径保留灰色
        
        for (cx, cy), radius in radii_info:
            # 绘制打断的黑洞范围（红色半透明效果）
            overlay = vis.copy()
            cv2.circle(overlay, (cx, cy), radius, (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
            
        cv2.imwrite(os.path.join(self.debug_dir, "05_erased_paths.png"), vis)
    def vis_topology(self, skeleton_img, endpoints, raw_intersections):
        """可视化：展示端点与密集的原始交叉点"""
        vis = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
        vis[skeleton_img == 255] = [200, 200, 200]
        if endpoints is not None:
            for y, x in endpoints:
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1) # 绿点：端点
        if raw_intersections is not None:
            for y, x in raw_intersections:
                cv2.circle(vis, (x, y), 2, (0, 0, 255), -1) # 红点：原始交叉像素
        
        cv2.imwrite(os.path.join(self.debug_dir, "02_raw_topology.png"), vis)

    def extractInts(self, skel):
        skel = skel.copy()
        skel[skel != 0] = 1
        skel = np.uint8(skel)
        kernel = np.uint8([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])
        src_depth = -1
        filtered = cv2.filter2D(skel, src_depth, kernel)

        p_ints = np.where(filtered > 12)

        return np.array([p_ints[0], p_ints[1]]) 
    def extractEndslist(self, skel):
        ends = self.extractEnds(skel)
        for e in ends:
            if e.shape[0] == 0:
                return []

        return list(zip(ends[1], ends[0]))

    def extractEnds(self, skel):

        skel = skel.copy()
        skel[skel != 0] = 1
        skel = np.uint8(skel)

        kernel = np.uint8([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])

        src_depth = -1
        filtered = cv2.filter2D(skel, src_depth, kernel)

        p_ends = np.where(filtered == 11)

        return np.array([p_ends[0], p_ends[1]])
    
    
    def adaptive_cluster_intersections(self, binary_mask, raw_intersections, dist_img, overlap_factor=1.2):
        """
        步骤 3 (增强版)：迭代式自适应重叠球聚类，彻底解决锐角导致的长 H 型遗留路径问题
        :param overlap_factor: 半径重叠宽容度。1.0 表示圆边刚好碰到就合并，1.5 表示圆之间相隔一段距离(长H杠)也会被强制吸附。
        """
        print("\n--- Running Step 3: Adaptive Overlap-Sphere Clustering ---")
        
        H, W = binary_mask.shape
        # 保底阈值 (防止线极细导致半径为0时不聚类)
        base_kernel_size = max(self.kernel_size, max(H,W) // 200)
        # 2. 初始化所有独立的“微型空洞”
        clusters = []
        for p in raw_intersections:
            py, px = p[0], p[1]
            # 获取该像素点的物理粗细半径
            pr = dist_img[py, px]
            pr = max(pr, 2.0) # 赋予一个极小的保底半径
            clusters.append({
                'point': np.array([py, px], dtype=float),
                'int_radius': float(pr)
            })
            
        # 3. 核心：迭代式融合，直到没有任何两个黑洞相交
        changed = True
        while changed:
            changed = False
            new_clusters = []
            merged_indices = set()
            for i in range(len(clusters)):
                if i in merged_indices: continue
                c1 = clusters[i]
                merged_this_round = False
                
                for j in range(i + 1, len(clusters)):
                    if j in merged_indices: continue
                    c2 = clusters[j]
                    
                    # 计算两个空洞中心的距离
                    dist_12 = np.hypot(c1['point'][0] - c2['point'][0], 
                                       c1['point'][1] - c2['point'][1])
                    
                    # 【自适应判定条件】：距离是否小于 (两圆半径之和 * 宽容度) 
                    # 只要 overlap_factor 设得当，哪怕长 H 型两端隔得很远，只要它们的势力范围接近，就会触发融合！
                    adaptive_thresh = max(base_kernel_size, (c1['int_radius'] + c2['int_radius']) * overlap_factor)
                    
                    if dist_12 < adaptive_thresh:
                        # ===== 触发融合 =====
                        # 取中点作为新的超级黑洞中心
                        new_center = (c1['point'] + c2['point']) / 2.0
                        
                        # 新的半径必须把 c1、c2 和中间的缝隙(遗留路径)全部吞噬！
                        # 新半径 = 两圆半径和 + 两点距离 的一半
                        new_radius = (c1['int_radius'] + c2['int_radius'] + dist_12) / 2.0
                        
                        new_clusters.append({
                            'point': new_center,
                            'int_radius': new_radius
                        })
                        
                        merged_indices.add(i)
                        merged_indices.add(j)
                        merged_this_round = True
                        changed = True
                        break # 发生融合后立刻跳出内圈，重新整理列表进行下一轮迭代
                
                if not merged_this_round:
                    new_clusters.append(c1)
                    
            clusters = new_clusters # 用融合后的列表覆盖，继续 while 循环

        # 4. 格式化输出为字典，无缝接入后续的代码
        ints_dict = {}
        for idx, c in enumerate(clusters):
            ints_dict[idx] = {
                "point": (int(round(c['point'][0])), int(round(c['point'][1]))),
                "int_radius": c['int_radius'],
                "routes_label": [], 
                "int_ends": []      
            }
            
        merged_centers = [d['point'] for d in ints_dict.values()]
        print(f"自适应聚类完成：从 {len(raw_intersections)} 个畸变像素合并为 {len(ints_dict)} 个超级黑洞。")
        
        return ints_dict, merged_centers, base_kernel_size
    
    def traverse_skeleton(self, sk, curr_pixel):
        path = [curr_pixel]
        while True:
            x, y = curr_pixel
            sk[x, y] = 0

            view = sk[x-1:x+2, y-1:y+2]
            if view[0, 0]: curr_pixel =   np.array([0, 0], dtype=np.int16)
            elif view[0, 1]: curr_pixel = np.array([0, 1], dtype=np.int16)
            elif view[0, 2]: curr_pixel = np.array([0, 2], dtype=np.int16)
            elif view[1, 0]: curr_pixel = np.array([1, 0], dtype=np.int16)
            elif view[1, 2]: curr_pixel = np.array([1, 2], dtype=np.int16)
            elif view[2, 0]: curr_pixel = np.array([2, 0], dtype=np.int16)
            elif view[2, 1]: curr_pixel = np.array([2, 1], dtype=np.int16)
            elif view[2, 2]: curr_pixel = np.array([2, 2], dtype=np.int16)
            else: curr_pixel = np.array([-1, -1], dtype=np.int16)

            if curr_pixel[0] != -1:
                curr_pixel[0] += x-1
                curr_pixel[1] += y-1
                path.append(curr_pixel)
            else:
                break

        path_len = len(path)
        np_path = np.zeros((path_len, 2), dtype=np.int16)
        for i in range(path_len):
            np_path[i] = path[i]
        return np_path
    def extractRoutes(self, ends, num_labels, labels, skel_img, ints_dict_rf):
        skel = skel_img.astype(bool)
        routes = {}
        ends_rf = []
        for n in range(1, num_labels + 1):
            ends_f = [e for e in ends if labels[tuple([e[1], e[0]])] == n]
            if len(ends_f) == 2:
                curr_pixel = np.array([ends_f[0][1], ends_f[0][0]]).astype('int16')
                route = self.traverse_skeleton(skel, curr_pixel)
                if len(route) > 0:
                    ends_rf += ends_f
                    rl_conn = []
                    for int_dict in ints_dict_rf.values():
                        if n in int_dict['routes_label']:
                            rl_conn.append(len(int_dict['routes_label']))
                    routes[n] = {'route': route, 'ends': [], 'ends_p': [], 'rl_conn': rl_conn}
            else:
                labels[labels==n] = 0
        return routes, labels, ends_rf
    
    def estimateRoutewidthFromSegment(self, routes, dist_img, min_px = 3):
        for i in routes.keys():
            widths = [round(dist_img[tuple(p)]) for p in routes[i]['route']]
            # widths_int = [np.round(dist_img[tuple(p)]) for p in routes[i]['route']]
            avg_width = np.mean(widths)
            max_width = np.max(widths)
            routes[i]['width'] = (avg_width, max_width)
        self.total_mean_width = np.mean([max(routes[i]['width'][0], self.kernel_size) for i in routes.keys()])
        return routes
    
    def constructEndsDict_New(self, routes, routes_im, ends, ints_dict):
        """
        构建端点字典，并完成 端点(Ends) - 路线(Routes) - 交叉点(Ints) 的三向拓扑绑定
        :param routes: 路线字典
        :param routes_im: 连通域标签图 (用于查询像素属于哪条路线)
        :param ends: 端点坐标列表 [(x, y), ...]
        :param ints_dict: 交叉点字典
        :return: ends_dict, ints_dict
        """
        ends_dict = {}
        # ==========================================
        # Step 1: 初始化端点，并将其挂载到对应的路线上
        # ==========================================
        for i, end_pt in enumerate(ends):
            ex, ey = end_pt[0], end_pt[1]
            
            # 从标签图读取该端点所属的路线 ID
            route_id = int(routes_im[ey, ex]) 
            
            # 读取物理线宽 (如果 route_id 存在且包含 width)
            # 兼容你之前的 width=(avg, max) 元组结构
            if route_id in routes and 'width' in routes[route_id]:
                end_radius = routes[route_id]['width'][0]
            else:
                end_radius = 0.0

            ends_dict[i] = {
                "point": (ex, ey),
                "route_label": route_id,
                "point_label": i,
                "point_type": 'iso', # 默认是孤立的(isolated)，后续判断如果连着交叉点则改为 'int'
                "pair_ends": [],
                "end_radius": end_radius
            }
            # 将端点 ID 和 坐标 反向记录到路线字典中
            if route_id in routes:
                if 'ends' not in routes[route_id]:
                    routes[route_id]['ends'] = []
                if 'ends_p' not in routes[route_id]:
                    routes[route_id]['ends_p'] = []
                    
                routes[route_id]['ends'].append(i)
                routes[route_id]['ends_p'].append((ex, ey))
        
        # ==========================================
        # Step 2: 建立 交叉点 -> 端点 的关联 (基于几何距离)
        # ==========================================
        for j, int_dict in ints_dict.items():
            # 获取交叉点中心 (py, px)
            p0 = int_dict['point']
            
            # 重新初始化route_label列表，准备记录与该交叉点关联的路线 ID
            # int_dict['route_label'] = []
            if 'int_ends' not in int_dict:
                int_dict['int_ends'] = []
            
            # 遍历所有刚生成的端点
            for e_idx, e_data in ends_dict.items():
                p1 = (e_data['point'][1], e_data['point'][0]) # 转为 (y, x) 以便计算距离
                
                # 计算端点到交叉点中心的物理距离
                dist = self.distance2D(p0, p1)
                if dist < int_dict['int_radius'] + self.kernel_size * 2:
                    # 1. 这个端点属于这个交叉点
                    e_data['point_type'] = 'int'
                    int_dict['int_ends'].append(e_idx)
                    #  # 2. 这条路径也属于这个交叉点
                    # rid = e_data['route_label']
                    # if rid not in int_dict['route_label']:
                    #     int_dict['route_label'].append(rid)
        return ends_dict, ints_dict
    
    def distance2D(self, point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
    
    
    def get_macro_vector(self, end_id, ends_dict, routes, stride=15):
        """
        沿着路线向后回退 stride 个像素，获取抗锯齿的真实宏观方向向量。
        向量方向：从路线内部 指向 交叉点 (Outward pointing)
        """
        end_data = ends_dict[end_id]
        route_id = end_data['route_label']
        ex, ey = end_data['point']
        
        # 兼容你的 route 像素列表获取，通常是 (y, x) 格式
        # 取出完整的路径像素点序列
        route_pixels = routes[route_id].get('route', routes[route_id].get('route_pixels', []))
        
        if len(route_pixels) < 2:
            return np.array([0.0, 0.0]) # 路线太短，无法计算向量
            
        # 找到当前端点在 route_pixels 序列中的位置 (是起点还是终点？)
        # 注意：route_pixels 通常是 (y, x)，而 end 是 (x, y)
        dist_to_start = np.hypot(ex - route_pixels[0][1], ey - route_pixels[0][0])
        dist_to_end = np.hypot(ex - route_pixels[-1][1], ey - route_pixels[-1][0])
        
        # 实际可用步长 (防止路线长度小于设定步长)
        actual_stride = min(stride, len(route_pixels) - 1)
        
        if dist_to_start < dist_to_end:
            # 端点在 route 的起点 [0]，往里走应该是下标 [actual_stride]
            macro_node = route_pixels[actual_stride]
        else:
            # 端点在 route 的终点 [-1]，往里走应该是下标 [-1 - actual_stride]
            macro_node = route_pixels[-1 - actual_stride]
            
        # 计算向量 (End - MacroNode)，统一用 (x, y) 表示向量
        vx = float(ex - macro_node[1])
        vy = float(ey - macro_node[0])
        
        # 归一化向量
        norm = np.hypot(vx, vy)
        if norm < 1e-6:
            return np.array([0.0, 0.0])
        return np.array([vx / norm, vy / norm])
    
    def compute_connection_cost(self, v1, v2):
        """
        计算两个端点连接的几何代价。
        因为向量 v1 和 v2 都是从各自的路线指向交叉点的，
        所以如果它们是完美的同一根直线，v1 和 v2 应该完全共线且方向相反 (夹角 180度)。
        因此，(v1 dot v2) 接近 -1 是最完美的。
        我们定义 Cost = 1 + (v1 dot v2)，范围在 [0, 2]。0 代表最平滑完美。
        """
        # 纯几何方向代价
        cos_theta = np.dot(v1, v2)
        # 加入极小的保底常数防止浮点数精度问题
        cost = 1.0 + cos_theta 
        return max(0.0, float(cost))
    
    def execute_local_combinatorial_matching(self, ints_dict, ends_dict, routes):
        """
        遍历所有交叉点，计算几何代价并返回最终配对的端点ID对。
        :return: matched_pairs_list -> [(e_id1, e_id2), (e_id3, e_id4), ...]
        """
        print("\n--- Running Step: Local Combinatorial Matching (Returning Pairs) ---")
        
        all_matched_pairs = []
        
        # 预先计算所有交叉点端点的宏观方向向量 (stride=15)
        vectors = {}
        for end_id, e_data in ends_dict.items():
            if e_data['point_type'] == 'int':
                vectors[end_id] = self.get_macro_vector(end_id, ends_dict, routes, stride=15)
                
        for int_id, int_data in ints_dict.items():
            local_ends = int_data.get('int_ends', [])
            num_ends = len(local_ends)
            if num_ends < 2:
                continue
                
            elif num_ends == 2:
                # 情况A: 2个端点，唯一配对
                all_matched_pairs.append((local_ends[0], local_ends[1]))
                
            elif num_ends == 3:
                # 情况B: 3个端点 (Y型)，找代价最低（最直）的一对
                best_pair = None
                min_cost = float('inf')
                for e1, e2 in itertools.combinations(local_ends, 2):
                    cost = self.compute_connection_cost(vectors[e1], vectors[e2])
                    if cost < min_cost:
                        min_cost = cost
                        best_pair = (e1, e2)
                if best_pair:
                    all_matched_pairs.append(best_pair)
                
            elif num_ends == 4:
                # 情况C: 4个端点 (X型)，排列组合 3 种拓扑
                topologies = [
                    [(local_ends[0], local_ends[1]), (local_ends[2], local_ends[3])],
                    [(local_ends[0], local_ends[2]), (local_ends[1], local_ends[3])],
                    [(local_ends[0], local_ends[3]), (local_ends[1], local_ends[2])]
                ]
                best_topo = None
                min_total_cost = float('inf')
                
                for topo in topologies:
                    c1 = self.compute_connection_cost(vectors[topo[0][0]], vectors[topo[0][1]])
                    c2 = self.compute_connection_cost(vectors[topo[1][0]], vectors[topo[1][1]])
                    if (c1 + c2) < min_total_cost:
                        min_total_cost = c1 + c2
                        best_topo = topo
                
                if best_topo:
                    all_matched_pairs.extend(best_topo)
                    
            else:
                # 情况D: >4 个端点 (复杂节点)，采用贪心算法
                unmatched = list(local_ends)
                while len(unmatched) >= 2:
                    best_pair = None
                    min_cost = float('inf')
                    for e1, e2 in itertools.combinations(unmatched, 2):
                        cost = self.compute_connection_cost(vectors[e1], vectors[e2])
                        if cost < min_cost:
                            min_cost = cost
                            best_pair = (e1, e2)
                    all_matched_pairs.append(best_pair)
                    unmatched.remove(best_pair[0])
                    unmatched.remove(best_pair[1])

        print(f"--- Matching Completed. Total Pairs Found: {len(all_matched_pairs)} ---")
        return all_matched_pairs

    def mergeEnds(self, skel, ends_dict_rf, end_pairs):
        skel_ = skel.copy()
        for end_pair in end_pairs:
            end_dict_1 = ends_dict_rf[end_pair[0]]
            end_dict_2 = ends_dict_rf[end_pair[1]]
            end_p1 = end_dict_1['point']
            end_p2 = end_dict_2['point']
            cv2.line(skel_, (end_p1[0], end_p1[1]), (end_p2[0], end_p2[1]), 255, thickness=1)
        return skel_
    
    
    
if __name__ == "__main__":
    # 请确保同级目录下有一张你的 HarnessHRNetV2 输出的二值化图片
    parser = CVFDLO_Parser(debug_dir="mydlo_debug", if_debug=True)
    parser.run("../final_inference_mask.png")