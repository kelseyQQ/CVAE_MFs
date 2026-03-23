# (Optional) mitigate OpenMP duplicate on Windows; set before imports that trigger MKL/OMP
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # only if you still hit libiomp conflicts
import numpy as np
import pandas as pd
import torch
from collections import deque
from CNN_CVAE import CVAE_encoder
import New_Minkowski_Dataset as md
from skimage import filters
from scipy.ndimage import gaussian_filter
from skimage.measure import euler_number, perimeter_crofton
import plotly.graph_objects as go
import pyvista as pv
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CONFIG = dict(
    runs_root=ROOT,
    ckpt=ROOT / "exploratory_3D" / "model_best.pth",    # relative to runs_root OR absolute path
    data_root=ROOT / "DAT_files_ori",
    latent=17,
    kernel_size=3,
    num_samples=400,
    random_n=100,                              # number of random targets
    bounds_ref="conditions",                   # 'conditions' | 'all_data'
    margin=0.0,                                # expand bounds by this fraction
    out=ROOT / "exploratory_3D" / "inverse_boundary.png",
    infer_shape=False,                         # set True to auto-detect latent (C,H,W)
    seed=0,
    norm_mode="range",                         # 'none'|'range'|'std'|'iqr'|'relative'
    norm_ref="conditions",                     # 'conditions'|'both'
    extra_views=[(10, 0), (10, 90)]
)

BIN_CFG = dict(sigma=0.2)
data_root = Path(CONFIG["data_root"])
folders_whole = [str(p) for p in data_root.iterdir() if p.is_dir()]
dataset_whole = md.MultiFolderMinkowskiDataset6(folders_whole)
global_mfs_50 = [dataset_whole[i][2] for i in range(0, len(dataset_whole), 400)]
cond50 = torch.stack(global_mfs_50).cpu().numpy()  # [50,3]

# === Global scale：all distances will use this single scale for range-normalization ===
# Here the reference for global scaling is cond50; you can change it to the whole dataset if you prefer.
REF = cond50
eps = 1e-12
GLOBAL_SCALE = np.maximum(REF.max(axis=0) - REF.min(axis=0), eps)

def norm_with_global_scale(conditions, means, *_, **__):
    """
    Use the same GLOBAL_SCALE for per-axis range scaling.
    *_,**__ is just to be compatible with the old calling convention (mode=..., ref=...), you can ignore those parameters.
    the first returned value is the scaled deltas, and the second is an info string (you can ignore it).
    """
    conditions = np.asarray(conditions, float)
    means = np.asarray(means, float)
    deltas = means - conditions           # Δ = μ - c
    scaled = deltas / GLOBAL_SCALE        # per-axis range scaling
    return scaled, "global-range L2 (per-axis)"

@torch.no_grad()
def infer_latent_shape(model, sample_img, sample_cond, device="cuda"):
    model.eval()
    x = sample_img.unsqueeze(0).to(device)      # [1,1,H,W]
    c = sample_cond.unsqueeze(0).to(device)     # [1,3]
    _, mu, _ = model(x, c)                      # mu: [1,C,H,W]
    _, C, H, W = mu.shape
    return (C, H, W)

def binarize(img, sigma=0.8):
    smooth = gaussian_filter(img, sigma=sigma)
    thresh = filters.threshold_otsu(smooth)
    bin_img = smooth >= thresh
    return bin_img.astype(bool)

def calculate_minkowski_functionals(binary_image):
    v0 = np.sum(binary_image) / binary_image.size                         # area fraction
    v1 = perimeter_crofton(binary_image, 4) / binary_image.shape[0]       # specific perimeter
    im_inv = np.logical_not(binary_image)
    v2 = euler_number(im_inv, connectivity=1)                             # Euler char of solid = χ(¬void)
    return np.array([v0, v1, v2], dtype=float)

@torch.no_grad()
def generate_means_for_conditions(model,
                                  conditions_np,
                                  latent_shape,
                                  device="cuda",
                                  num_samples=400,
                                  return_std=False):
    C, H, W = latent_shape
    model.eval()
    cond = torch.as_tensor(conditions_np, dtype=torch.float32, device=device)  # [N,3]
    N = cond.shape[0]
    mu_list = []
    std_list = []  

    for i in range(N):
        c = cond[i]                                 # [3]
        # Expand condition to spatial map
        cond_map = c.view(1, 3, 1, 1).expand(num_samples, 3, H, W)  # [S,3,H,W]
        z = torch.randn(num_samples, C, H, W, device=device)        # [S,C,H,W]
        dec_in = torch.cat([z, cond_map], dim=1)                    # [S,C+3,H,W]
        recon_logits = model.decoder(dec_in)                     
        recon = torch.sigmoid(recon_logits).squeeze(1).detach().cpu().numpy()  # [S,H',W']

        mfs = []
        for k in range(num_samples):
            bin_img = binarize(recon[k], **BIN_CFG)
            mfs.append(calculate_minkowski_functionals(bin_img))
        mfs = np.stack(mfs, axis=0)          # [S,3]
        mu = mfs.mean(axis=0)                # [3]
        mu_list.append(mu)
        if return_std:
            std_list.append(mfs.std(axis=0, ddof=1))

    mu_all = np.stack(mu_list, axis=0)       # [N,3]
    if return_std:
        std_all = np.stack(std_list, axis=0)              # [N,3]
        return mu_all, std_all
    return mu_all

def eval_point(pt_norm, mn, scale, model, latent_shape, gen_fn, norm_fn, device, num_samples):
    """Evaluate error at a single normalized point."""
    target_raw = mn + pt_norm * scale
    if not is_valid_mf_raw(target_raw):
        return np.inf, target_raw
    conds = target_raw[None, :]
    means = gen_fn(model, conds, latent_shape, device=device, num_samples=num_samples)
    scaled_deltas, _ = norm_fn(conds, means)
    dist = float(np.linalg.norm(scaled_deltas[0]))
    return dist, target_raw

def is_valid_mf_raw(P):
    """
    P: (..., 3) raw Minkowski functionals [M0, M1, M2]
    valid if: 0 < M0 < 1 and M1 > 0
    """
    P = np.asarray(P, dtype=float)
    m0 = P[..., 0]
    m1 = P[..., 1]
    return (m0 > 0.0) & (m0 < 1.0) & (m1 > 0.0)

def octree_boundary_corner_points(
    cond50,
    model,
    latent_shape,
    gen_fn,
    norm_fn,
    threshold=0.25,      
    max_depth=3,         
    min_half=0.0,        
    expand=0.5,          
    device="cuda",
    num_samples=150,
    cache_round=12,     
):
    cond50 = np.asarray(cond50, float)
    mn = cond50.min(axis=0)
    mx = cond50.max(axis=0)
    scale = np.maximum(mx - mn, 1e-12)

    # normalized domain
    cond_norm = (cond50 - mn) / scale
    dom_min, dom_max = cond_norm.min(axis=0), cond_norm.max(axis=0)

    center = 0.5 * (dom_min + dom_max)
    half = 0.5 * float(np.max(dom_max - dom_min)) * (1.0 + float(expand))

    # --- corner evaluation cache ---
    # key: tuple( round(x,cache_round), round(y,...), round(z,...) ) in normalized space
    eval_cache = {}

    def eval_corner(pt_norm):
        key = tuple(np.round(pt_norm, cache_round))
        if key in eval_cache:
            return eval_cache[key]

        dist, pt_raw = eval_point(
            pt_norm=pt_norm,
            mn=mn,
            scale=scale,
            model=model,
            latent_shape=latent_shape,
            gen_fn=gen_fn,
            norm_fn=norm_fn,
            device=device,
            num_samples=num_samples,
        )
        eval_cache[key] = (float(dist), np.asarray(pt_raw, float))
        return eval_cache[key]

    # --- octree queue ---
    queue = deque()
    queue.append((center, half, 0))  # (center_norm, half_size, depth)

    # --- output unique points (raw space) ---
    # key in raw space rounded to avoid float duplicates across adjacent cells
    out = {}  # key_raw -> dict(x,y,z,dist,color)

    corner_offsets = np.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ], dtype=float)

    while queue:
        c, h, depth = queue.popleft()

        # 8 corners in normalized space
        corners_norm = c[None, :] + h * corner_offsets  # (8,3)

        # Evaluate 8 corners
        # Evaluate corners (but ignore raw M0 <= 0 BEFORE calling eval_corner)
        corners_raw_geom = mn + corners_norm * scale   # (8,3)

        # valid if 0 < M0 < 1 and M1 > 0
        valid = is_valid_mf_raw(corners_raw_geom)# raw porosity (M0) > 0 

        if not np.any(valid):
            continue

        dists = np.full((8,), np.nan, dtype=float)
        corners_raw = np.zeros((8,3), dtype=float)
        for i in np.where(valid)[0]:
            d, pr = eval_corner(corners_norm[i])
            dists[i] = d
            corners_raw[i] = pr

        d_valid = dists[valid]
        d_min = float(d_valid.min())
        d_max = float(d_valid.max())


        all_outside = d_min > threshold          # all > threshold
        all_inside  = d_max <= threshold         # all <= threshold
        is_boundary = (d_min <= threshold) and (d_max > threshold)

        FORCE_REFINE_LEVELS = 4  
        if all_outside and depth >= FORCE_REFINE_LEVELS:
            continue

        if all_inside:
            continue

        is_leaf = (depth >= max_depth) or (min_half > 0.0 and h <= min_half)

        if (not is_leaf):
            child_half = h * 0.5
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        child_center = c + child_half * np.array([sx, sy, sz], dtype=float)
                        queue.append((child_center, child_half, depth + 1))
            continue
            
        if is_boundary:
            green_mask = valid & (dists <= threshold)
            red_mask   = valid & (dists >  threshold)

            has_green = bool(np.any(green_mask))
            has_red   = bool(np.any(red_mask))

            if has_green and has_red:
                # store cube center (raw space)
                center_raw = mn + c * scale   # c is center in normalized space
                if not is_valid_mf_raw(center_raw):
                    continue
                x, y, z = center_raw.tolist()

                d_valid = dists[valid]
                d_min = float(np.nanmin(d_valid))
                d_max = float(np.nanmax(d_valid))

                key_raw = (round(x, 12), round(y, 12), round(z, 12))
                score = d_max - d_min
                if key_raw not in out or score > out[key_raw].get("score", -1.0):
                    out[key_raw] = {
                        "x": x, "y": y, "z": z,
                        "d_min": d_min, "d_max": d_max,
                        "score": score,
                        "color": "center_mixed"
                    }

    if len(out) == 0:
        meta_df = pd.DataFrame(columns=["x", "y", "z", "dist", "color"])
        meta_df = pd.DataFrame(columns=["x", "y", "z", "d_min", "d_max", "score", "color"])
        center_pts_raw = np.zeros((0,3), dtype=float)
        return center_pts_raw, meta_df

    meta_df = pd.DataFrame(list(out.values()))
    if "d_min" in meta_df.columns:
        meta_df = meta_df.sort_values(["d_min"], ascending=[True]).reset_index(drop=True)
    else:
        meta_df = meta_df.reset_index(drop=True)

    center_pts_raw = meta_df[["x","y","z"]].to_numpy(dtype=float)
    return center_pts_raw, meta_df

def build_model(latent_channels: int, kernel_size: int, device="cuda"):
    return CVAE_encoder(
        channel_in=1, ch=16, blocks=(1,2,4),
        latent_channels=latent_channels,
        num_res_blocks=1, norm_type="bn",
        deep_model=False, condition_dim=3, kernel_size=kernel_size
    ).to(device)

def set_global_seed(seed: int = 0):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(CONFIG["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model(CONFIG["latent"], CONFIG["kernel_size"], device=device)
ckpt_path = Path(CONFIG["ckpt"])
try:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
except TypeError:
    state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state, strict=False)
model.eval()

if CONFIG["infer_shape"]:
    img0, _, cond0, _ = dataset_whole[0]
    latent_shape = infer_latent_shape(model, img0, cond0, device=device)
else:
    latent_shape = (CONFIG["latent"], 5, 5) 


if __name__ == "__main__":
    THR_A = 0.25
    THR_B = 0.5
    MAX_DEPTH = 5

    out_dir = os.path.join(CONFIG["runs_root"], "exploratory_3D", "octree_pointcloud")
    os.makedirs(out_dir, exist_ok=True)

    def run_octree_and_surface(thr, alpha_factor=0.08):
        center_pts_raw, meta_df = octree_boundary_corner_points(
            cond50=cond50,
            model=model,
            latent_shape=latent_shape,
            gen_fn=generate_means_for_conditions,
            norm_fn=norm_with_global_scale,
            threshold=thr,
            max_depth=MAX_DEPTH,
            expand=1.5,
            device=device,
            num_samples=CONFIG["num_samples"],
        )

        P = np.asarray(center_pts_raw, dtype=float)
        if P.shape[0] < 4:
            print(f"[thr={thr}] Not enough center points to form a surface: {P.shape[0]}")
            return None, None, None

        mn = cond50.min(axis=0)
        mx = cond50.max(axis=0)
        scale = np.maximum(mx - mn, 1e-12)

        P_norm = (P - mn) / scale
        cloud = pv.PolyData(P_norm)

        xmin, xmax, ymin, ymax, zmin, zmax = cloud.bounds
        diag = ((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2) ** 0.5
        alpha = alpha_factor * diag

        tetra = cloud.delaunay_3d(alpha=alpha)
        surf = tetra.extract_surface().triangulate()

        # save STL (per threshold)
        alpha_tag = f"{alpha:.3f}"
        out_surface = os.path.join(out_dir, f"green_surface_thr_{thr}_alpha_{alpha_tag}.stl")
        surf.save(out_surface)
        print(f"[thr={thr}] saved surface:", out_surface)

        return surf, alpha, (mn, scale)

    # ---- run both thresholds ----
    surf_a, alpha_a, norm_pack = run_octree_and_surface(THR_A, alpha_factor=0.08)
    surf_b, alpha_b, _         = run_octree_and_surface(THR_B, alpha_factor=0.08)

    # ---- combined preview: two surfaces + cond50 points ----
    if norm_pack is None:
        raise RuntimeError("Failed to compute normalization pack (mn, scale).")

    mn, scale = norm_pack
    cond50_norm = (cond50 - mn) / scale
    cond50_cloud = pv.PolyData(cond50_norm)

    p = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    p.set_background("white")

    # thr=0.25: less transparent
    if surf_a is not None and surf_a.n_points > 0 and surf_a.n_cells > 0:
        p.add_mesh(
            surf_a,
            color="seagreen",     
            opacity=0.5,
            show_edges=False
        )
    # thr=0.8: more transparent 
    if surf_b is not None and surf_b.n_points > 0 and surf_b.n_cells > 0:
        p.add_mesh(
            surf_b,
            color="royalblue",    
            opacity=0.15,
            show_edges=False
        )
    p.add_points(
        cond50_cloud,
        color="black",
        point_size=10,
        render_points_as_spheres=True,
    )

    p.show_axes()
    p.show_grid()
    p.reset_camera()

    png_path = os.path.join(out_dir, f"preview_thr_{THR_A}_and_{THR_B}.png")
    p.camera_position = "iso"  

    # p.camera_position = [
    #     (2.0, 2.0, 1.6),  # camera location
    #     (0.5, 0.5, 0.5),  # focal point
    #     (0.0, 0.0, 1.0)   # view up
    # ]
    p.show(screenshot=png_path, auto_close=True)
    print("saved combined preview:", png_path)


    # =========================
    # Export interactive HTML (Plotly)
    # =========================
    def pv_surface_to_mesh3d(surf, name, color="blue",opacity=0.5):
        """
        Convert a triangulated pyvista PolyData surface to a Plotly Mesh3d trace.
        Assumes surf is in normalized coordinates already.
        """
        if surf is None or surf.n_points == 0 or surf.n_cells == 0:
            return None

        V = np.asarray(surf.points)
        # faces are stored like: [3, v0, v1, v2, 3, v0, v1, v2, ...]
        F = np.asarray(surf.faces).reshape(-1, 4)[:, 1:4]

        return go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=F[:, 0], j=F[:, 1], k=F[:, 2],
            color=color,
            opacity=opacity,
            name=name,
            showscale=False,
        )

    fig = go.Figure()

    # two surfaces
    trace_a = pv_surface_to_mesh3d(surf_a, name=f"surface thr={THR_A}",color="seagreen", opacity=0.5)
    if trace_a is not None:
        fig.add_trace(trace_a)

    trace_b = pv_surface_to_mesh3d(surf_b, name=f"surface thr={THR_B}",color="cornflowerblue", opacity=0.15)
    if trace_b is not None:
        fig.add_trace(trace_b)

    # cond50 points
    fig.add_trace(go.Scatter3d(
        x=cond50_norm[:, 0], y=cond50_norm[:, 1], z=cond50_norm[:, 2],
        mode="markers",
        marker=dict(size=3,color="black"),
        name="cond50",
    ))

    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Two surfaces (center points) + cond50 | thr={THR_A} & {THR_B}",
        legend=dict(itemsizing="constant"),
    )

    def _get_norm_range_from_fig(fig):
        xs, ys, zs = [], [], []
        for tr in fig.data:
            if getattr(tr, "x", None) is None:
                continue
            xs.append(np.asarray(tr.x, dtype=float))
            ys.append(np.asarray(tr.y, dtype=float))
            zs.append(np.asarray(tr.z, dtype=float))
        X = np.concatenate(xs) if xs else np.array([0.0, 1.0])
        Y = np.concatenate(ys) if ys else np.array([0.0, 1.0])
        Z = np.concatenate(zs) if zs else np.array([0.0, 1.0])

        def pad(a, frac=0.05):
            lo, hi = float(np.min(a)), float(np.max(a))
            r = hi - lo
            if r <= 1e-12:
                r = 1.0
            p = r * frac
            return lo - p, hi + p

        return pad(X), pad(Y), pad(Z)

    def _make_ticks_from_norm_range(norm_lo, norm_hi, mn, scale, n=7):
        tickvals = np.linspace(norm_lo, norm_hi, n)          # positions in normalized space
        ticktext = [f"{(mn + v*scale):.2f}" for v in tickvals]  # labels in raw space
        return tickvals.tolist(), ticktext

    # 1) get actual normalized range from the plotted data
    (xn_lo, xn_hi), (yn_lo, yn_hi), (zn_lo, zn_hi) = _get_norm_range_from_fig(fig)

    # 2) build ticks that cover that range, but display raw labels
    x_tickvals, x_ticktext = _make_ticks_from_norm_range(xn_lo, xn_hi, mn[0], scale[0], n=7)
    y_tickvals, y_ticktext = _make_ticks_from_norm_range(yn_lo, yn_hi, mn[1], scale[1], n=7)
    z_tickvals, z_ticktext = _make_ticks_from_norm_range(zn_lo, zn_hi, mn[2], scale[2], n=7)

    # 3) set axis range + ticks + titles
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="M0", range=[xn_lo, xn_hi], tickmode="array", tickvals=x_tickvals, ticktext=x_ticktext),
            yaxis=dict(title="M1", range=[yn_lo, yn_hi], tickmode="array", tickvals=y_tickvals, ticktext=y_ticktext),
            zaxis=dict(title="M2", range=[zn_lo, zn_hi], tickmode="array", tickvals=z_tickvals, ticktext=z_ticktext),
            aspectmode="data",
        )
    )
    html_path = os.path.join(out_dir, f"preview_thr_{THR_A}_and_{THR_B}.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print("saved interactive html:", html_path)
