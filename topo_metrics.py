"""
topo_metrics.py
================
Topology-aware metrics for centerline / tubular structure segmentation.

All metric functions follow the same convention:
    Input:
        pred_skel : np.ndarray  (H, W), single-pixel-wide skeleton, values in {0, 1}
        gt_skel   : np.ndarray  (H, W), single-pixel-wide skeleton, values in {0, 1}
    Output:
        dict[str, float]   — one or more named scalar values

Designed for low coupling: import only what you need; each function is independent.

References
----------
- Shit et al., "clDice - a Novel Topology-Preserving Loss Function for
  Tubular Structure Segmentation", CVPR 2021.
- Wiedemann et al., "Empirical evaluation of automatically extracted road axes", 1998.
- Van Etten et al., "The SpaceNet Roads Dataset", 2018  (APLS).
- Heimann et al., "Comparison and Evaluation of Methods for Liver Segmentation
  From CT Datasets", IEEE T-MI 2009  (HD95, ASSD).
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import DBSCAN
import networkx as nx


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_binary(arr: np.ndarray) -> np.ndarray:
    """Cast to uint8 {0,1} regardless of input dtype/scale."""
    if arr.dtype == bool:
        return arr.astype(np.uint8)
    return (arr > 0).astype(np.uint8)


def _empty_check(pred: np.ndarray, gt: np.ndarray) -> dict | None:
    """Return a sensible dict if either input is empty; else None to continue."""
    p_empty = pred.sum() == 0
    g_empty = gt.sum() == 0
    if p_empty and g_empty:
        return {"empty": True}
    return None


# ===========================================================================
# 1. clDice  —  centerline Dice  (Shit et al., CVPR 2021)
# ===========================================================================

def cldice(pred_skel: np.ndarray, gt_skel: np.ndarray) -> dict:
    """
    clDice =  2 * (Tprec * Tsens) / (Tprec + Tsens)
        Tprec = |pred_skel ∩ gt| / |pred_skel|
        Tsens = |gt_skel ∩ pred| / |gt_skel|

    For skeleton-vs-skeleton evaluation we use the skeletons themselves
    as both the "centerline" and the "mask" (degenerate but standard
    when GT is already a skeleton).
    """
    p = _ensure_binary(pred_skel).astype(bool)
    g = _ensure_binary(gt_skel).astype(bool)

    if p.sum() == 0 and g.sum() == 0:
        return {"clDice": 1.0}
    if p.sum() == 0 or g.sum() == 0:
        return {"clDice": 0.0}

    inter = np.logical_and(p, g).sum()
    tprec = (inter + 1e-7) / (p.sum() + 1e-7)
    tsens = (inter + 1e-7) / (g.sum() + 1e-7)
    cl = 2.0 * tprec * tsens / (tprec + tsens + 1e-7)
    return {"clDice": float(cl)}


# ===========================================================================
# 2. Relaxed Precision / Recall / F1 at multiple tolerances
# ===========================================================================

def relaxed_prf(pred_skel: np.ndarray,
                gt_skel: np.ndarray,
                tolerances=(2, 3, 5)) -> dict:
    """
    Buffer-based evaluation (Wiedemann et al., 1998).
    A predicted pixel counts as TP if it lies within `tol` pixels of any
    GT pixel; symmetric for recall.

    Returns a flat dict like {"P@2": ..., "R@2": ..., "F1@2": ..., "P@3": ...}.
    """
    p = _ensure_binary(pred_skel)
    g = _ensure_binary(gt_skel)

    out = {}
    if p.sum() == 0 and g.sum() == 0:
        for t in tolerances:
            out[f"P@{t}"] = out[f"R@{t}"] = out[f"F1@{t}"] = 1.0
        return out
    if p.sum() == 0 or g.sum() == 0:
        for t in tolerances:
            out[f"P@{t}"] = out[f"R@{t}"] = out[f"F1@{t}"] = 0.0
        return out

    # Distance from every pixel to the nearest non-zero pixel of the other mask
    gt_dist = cv2.distanceTransform(1 - g, cv2.DIST_L2, 3)
    pred_dist = cv2.distanceTransform(1 - p, cv2.DIST_L2, 3)

    for tol in tolerances:
        tp_p = np.sum((p > 0) & (gt_dist <= tol))
        precision = (tp_p + 1e-7) / (p.sum() + 1e-7)

        tp_r = np.sum((g > 0) & (pred_dist <= tol))
        recall = (tp_r + 1e-7) / (g.sum() + 1e-7)

        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        out[f"P@{tol}"] = float(precision)
        out[f"R@{tol}"] = float(recall)
        out[f"F1@{tol}"] = float(f1)
    return out


# ===========================================================================
# 3. HD95  &  ASSD   —  distance-based metrics
# ===========================================================================

def hd95_assd(pred_skel: np.ndarray, gt_skel: np.ndarray) -> dict:
    """
    HD95  = 95th percentile of the symmetric set of nearest-neighbour distances.
    ASSD  = Average Symmetric Surface Distance (mean of all those distances).

    These two together replace the noisy max-Hausdorff and give a clear
    picture of both worst-case and average geometric error.
    Reported in pixels (image space).
    """
    p = _ensure_binary(pred_skel)
    g = _ensure_binary(gt_skel)

    if p.sum() == 0 and g.sum() == 0:
        return {"HD95": 0.0, "ASSD": 0.0}
    if p.sum() == 0 or g.sum() == 0:
        # Worst-case: image diagonal
        diag = float(np.sqrt(p.shape[0] ** 2 + p.shape[1] ** 2))
        return {"HD95": diag, "ASSD": diag}

    # For each pred pixel, distance to nearest gt pixel, and vice versa
    dt_gt = cv2.distanceTransform(1 - g, cv2.DIST_L2, 3)
    dt_pred = cv2.distanceTransform(1 - p, cv2.DIST_L2, 3)

    d_p2g = dt_gt[p > 0]          # pred -> gt
    d_g2p = dt_pred[g > 0]        # gt -> pred

    all_d = np.concatenate([d_p2g, d_g2p])
    hd95 = float(np.percentile(all_d, 95))
    assd = float(all_d.mean())
    return {"HD95": hd95, "ASSD": assd}


# ===========================================================================
# 4. Betti number errors  (β0 = #components, β1 = #loops)
# ===========================================================================

def betti_errors(pred_skel: np.ndarray, gt_skel: np.ndarray) -> dict:
    """
    Topology-level error: how many connected components and how many loops
    differ between prediction and GT?

    β0  =  number of connected components
    β1  =  number of independent loops
    For 2D binary images, Euler characteristic  χ = β0 − β1,
    and χ can be computed exactly from the pixel grid.

    We use 8-connectivity for β0 (standard for skeletons).
    """
    p = _ensure_binary(pred_skel)
    g = _ensure_binary(gt_skel)

    def _betti(binary):
        if binary.sum() == 0:
            return 0, 0
        # β0: connected components (8-connectivity)
        struct = ndimage.generate_binary_structure(2, 2)  # 8-conn
        _, b0 = ndimage.label(binary, structure=struct)
        # Euler number via OpenCV's connectedComponents on background trick
        # χ = β0 − β1  ⇒  β1 = β0 − χ
        # Use scikit-image-style: χ = V − E + F for the cubical complex;
        # OpenCV doesn't expose it, but we can compute via background components.
        # Easier: number of holes = (background components with 4-conn) − 1
        struct_bg = ndimage.generate_binary_structure(2, 1)  # 4-conn for bg
        _, bg_components = ndimage.label(1 - binary, structure=struct_bg)
        # One of these is the unbounded outer region; rest are holes
        b1 = max(0, bg_components - 1)
        return b0, b1

    p_b0, p_b1 = _betti(p)
    g_b0, g_b1 = _betti(g)

    return {
        "Betti0_err": float(abs(p_b0 - g_b0)),
        "Betti1_err": float(abs(p_b1 - g_b1)),
        "pred_b0": int(p_b0),
        "gt_b0": int(g_b0),
        "pred_b1": int(p_b1),
        "gt_b1": int(g_b1),
    }


# ===========================================================================
# 5. APLS — Average Path Length Similarity  (SpaceNet Roads)
# ===========================================================================

def _skeleton_to_graph(skel: np.ndarray) -> nx.Graph:
    """
    Convert a single-pixel skeleton into a NetworkX graph where
        node    = junction or endpoint pixel
        edge    = a path between two such nodes, weighted by its pixel length.
    """
    G = nx.Graph()
    skel = _ensure_binary(skel)
    if skel.sum() == 0:
        return G

    # Find node pixels: endpoints (1 neighbour) and junctions (≥3 neighbours)
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    nb = cv2.filter2D(skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    nb = nb * skel  # only count where skel is present
    node_mask = (skel > 0) & ((nb == 1) | (nb >= 3))
    node_coords = np.argwhere(node_mask)

    if len(node_coords) == 0:
        # Pure loop: pick an arbitrary pixel as anchor
        ys, xs = np.where(skel > 0)
        node_coords = np.array([[ys[0], xs[0]]])
        node_mask[ys[0], xs[0]] = True

    node_id_map = {tuple(c): i for i, c in enumerate(node_coords)}
    for (r, c), i in node_id_map.items():
        G.add_node(i, pos=(r, c))

    # Walk each non-node skeleton pixel to find the edges between nodes.
    # For each node, do BFS along skeleton pixels until we hit another node.
    visited_edges = set()
    H, W = skel.shape

    def neighbours(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and skel[nr, nc]:
                    yield nr, nc

    for (r0, c0), nid in node_id_map.items():
        # BFS from this node along non-node skeleton
        for nr, nc in neighbours(r0, c0):
            # Walk this branch
            prev = (r0, c0)
            cur = (nr, nc)
            length = float(np.hypot(nr - r0, nc - c0))
            steps = 0
            while True:
                steps += 1
                if steps > H * W:   # safety
                    break
                if cur in node_id_map and cur != (r0, c0):
                    other = node_id_map[cur]
                    edge_key = tuple(sorted([nid, other]))
                    if edge_key not in visited_edges:
                        if nid != other:
                            G.add_edge(nid, other, weight=length)
                        visited_edges.add(edge_key)
                    break
                # find next non-prev neighbour
                nxts = [n for n in neighbours(*cur) if n != prev]
                if not nxts:
                    # Dead end without a node — happens at degree-2 endpoints
                    # already accounted for; just stop.
                    break
                nxt = nxts[0]
                length += float(np.hypot(nxt[0] - cur[0], nxt[1] - cur[1]))
                prev = cur
                cur = nxt
    return G


def _match_nodes(G_pred: nx.Graph, G_gt: nx.Graph, max_dist: float = 5.0):
    """Greedy spatial matching of nodes between two graphs by pixel distance."""
    pred_nodes = list(G_pred.nodes(data="pos"))
    gt_nodes = list(G_gt.nodes(data="pos"))
    if not pred_nodes or not gt_nodes:
        return {}

    pred_pos = np.array([p for _, p in pred_nodes])
    gt_pos = np.array([p for _, p in gt_nodes])
    pred_ids = [i for i, _ in pred_nodes]
    gt_ids = [i for i, _ in gt_nodes]

    # Cost matrix
    cost = np.linalg.norm(
        pred_pos[:, None, :] - gt_pos[None, :, :], axis=2
    )
    # Pad to square for Hungarian if needed; here we use rectangular linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = {}   # gt_id -> pred_id
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_dist:
            matches[gt_ids[c]] = pred_ids[r]
    return matches


def apls(pred_skel: np.ndarray,
         gt_skel: np.ndarray,
         max_node_dist: float = 5.0,
         max_paths: int = 500) -> dict:
    """
    Simplified APLS for skeleton graphs.

    For each pair of GT nodes (s, t):
        L_gt = shortest-path length in GT graph
        If both s and t have matched pred nodes (s', t'):
            L_pred = shortest-path length in pred graph between s' and t'
            score  = max(0, 1 − |L_pred − L_gt| / L_gt)
        Else:
            score = 0
    APLS = mean of all such scores.

    `max_paths` caps the number of node pairs sampled to keep it tractable
    on dense graphs.
    """
    p = _ensure_binary(pred_skel)
    g = _ensure_binary(gt_skel)

    if g.sum() == 0 and p.sum() == 0:
        return {"APLS": 1.0}
    if g.sum() == 0 or p.sum() == 0:
        return {"APLS": 0.0}

    G_pred = _skeleton_to_graph(p)
    G_gt = _skeleton_to_graph(g)

    if G_gt.number_of_nodes() < 2 or G_pred.number_of_nodes() < 1:
        return {"APLS": 0.0}

    matches = _match_nodes(G_pred, G_gt, max_dist=max_node_dist)

    gt_nodes = list(G_gt.nodes())
    rng = np.random.default_rng(0)
    if len(gt_nodes) * (len(gt_nodes) - 1) // 2 > max_paths:
        # Sample pairs
        pairs = []
        for _ in range(max_paths):
            a, b = rng.choice(len(gt_nodes), 2, replace=False)
            pairs.append((gt_nodes[a], gt_nodes[b]))
    else:
        pairs = [(gt_nodes[i], gt_nodes[j])
                 for i in range(len(gt_nodes))
                 for j in range(i + 1, len(gt_nodes))]

    scores = []
    for s, t in pairs:
        try:
            L_gt = nx.shortest_path_length(G_gt, s, t, weight="weight")
        except nx.NetworkXNoPath:
            continue
        if L_gt <= 0:
            continue

        if s in matches and t in matches:
            sp, tp = matches[s], matches[t]
            try:
                L_pred = nx.shortest_path_length(G_pred, sp, tp, weight="weight")
                score = max(0.0, 1.0 - abs(L_pred - L_gt) / L_gt)
            except nx.NetworkXNoPath:
                score = 0.0
        else:
            score = 0.0
        scores.append(score)

    if not scores:
        return {"APLS": 0.0}
    return {"APLS": float(np.mean(scores))}


# ===========================================================================
# 6. Junction (intersection) accuracy with Hungarian matching
# ===========================================================================

def junction_accuracy(pred_skel: np.ndarray,
                      gt_skel: np.ndarray,
                      tolerance: float = 5.0,
                      cluster_eps: float = 8.0) -> dict:
    """
    Detect junction pixels (≥3 neighbours in 3x3), cluster them with DBSCAN,
    then match predicted clusters to GT clusters with the Hungarian algorithm.

    Returns precision, recall, F1 of junction detection.
    """
    p = _ensure_binary(pred_skel)
    g = _ensure_binary(gt_skel)

    def _junctions(skel):
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0
        nb = cv2.filter2D(skel, -1, kernel) * skel
        pts = np.argwhere((skel > 0) & (nb >= 3))
        if len(pts) == 0:
            return np.zeros((0, 2))
        # Cluster nearby junction pixels (junctions often span a few pixels)
        db = DBSCAN(eps=cluster_eps, min_samples=1).fit(pts)
        centers = []
        for lab in set(db.labels_):
            mask = db.labels_ == lab
            centers.append(pts[mask].mean(axis=0))
        return np.array(centers)

    P = _junctions(p)
    G = _junctions(g)

    if len(P) == 0 and len(G) == 0:
        return {"Junction_P": 1.0, "Junction_R": 1.0, "Junction_F1": 1.0,
                "n_pred_junc": 0, "n_gt_junc": 0}
    if len(P) == 0:
        return {"Junction_P": 0.0, "Junction_R": 0.0, "Junction_F1": 0.0,
                "n_pred_junc": 0, "n_gt_junc": int(len(G))}
    if len(G) == 0:
        return {"Junction_P": 0.0, "Junction_R": 0.0, "Junction_F1": 0.0,
                "n_pred_junc": int(len(P)), "n_gt_junc": 0}

    cost = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    # Hungarian on a possibly rectangular cost
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = int(np.sum(cost[row_ind, col_ind] <= tolerance))

    precision = matched / len(P)
    recall = matched / len(G)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return {
        "Junction_P": float(precision),
        "Junction_R": float(recall),
        "Junction_F1": float(f1),
        "n_pred_junc": int(len(P)),
        "n_gt_junc": int(len(G)),
    }


# ===========================================================================
# 7. Composite "compute_all" — convenience wrapper
# ===========================================================================

def compute_all(pred_skel: np.ndarray,
                gt_skel: np.ndarray,
                tolerances=(2, 3, 5)) -> dict:
    """
    Run every metric and merge into a single flat dict.
    Order is fixed so CSV column order is reproducible.
    """
    out = {}
    out.update(cldice(pred_skel, gt_skel))
    out.update(relaxed_prf(pred_skel, gt_skel, tolerances=tolerances))
    out.update(hd95_assd(pred_skel, gt_skel))
    out.update(betti_errors(pred_skel, gt_skel))
    out.update(apls(pred_skel, gt_skel))
    out.update(junction_accuracy(pred_skel, gt_skel))
    return out


# A canonical ordered list of "headline" metric keys for nice tables
HEADLINE_METRICS = [
    "clDice", "F1@2", "F1@3", "F1@5",
    "HD95", "ASSD",
    "Betti0_err", "Betti1_err",
    "APLS", "Junction_F1",
]
