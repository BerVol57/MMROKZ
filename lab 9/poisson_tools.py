import numpy as np
import cv2
import cupy as cp
import cupyx
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import cg


def imread_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im.astype(np.float64)

def imwrite_rgb(path, arr):
    if hasattr(arr, 'get'):
        arr = arr.get()
    arru = np.clip(arr, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arru, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

# ---------------------------
# GPU Poisson Helpers
# ---------------------------
def _build_laplacian_matrix_gpu(n_pixels, mask_gpu):
    """
    Laplacian Sparse Matrix on GPU
    """
    h, w = mask_gpu.shape
    
    coords = cp.argwhere(mask_gpu)
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    
    idx_map = -cp.ones((h, w), dtype=cp.int32)
    idx_map[mask_gpu] = cp.arange(n_pixels, dtype=cp.int32)

    center_idx = idx_map[y_coords, x_coords]
    
    rows = []
    cols = []
    vals = []
    
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for dy, dx in neighbors:
        ny, nx = y_coords + dy, x_coords + dx
        
        valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        
        ny_v = ny[valid]
        nx_v = nx[valid]
        curr_v = center_idx[valid]
        
        is_in_mask = mask_gpu[ny_v, nx_v]
        
        neigh_idx = idx_map[ny_v, nx_v]
        
        rows.append(curr_v[is_in_mask])
        cols.append(neigh_idx[is_in_mask])
        vals.append(cp.full(int(cp.sum(is_in_mask)), -1.0, dtype=cp.float64))

    # Add Diagonal (4.0)
    rows.append(center_idx)
    cols.append(center_idx)
    vals.append(cp.full(n_pixels, 4.0, dtype=cp.float64))
    
    rows_gpu = cp.concatenate(rows)
    cols_gpu = cp.concatenate(cols)
    vals_gpu = cp.concatenate(vals)
    
    A = csp.csr_matrix((vals_gpu, (rows_gpu, cols_gpu)), shape=(n_pixels, n_pixels))
    return A, idx_map

def _compute_gradients_gpu(img_gpu):
    grad_x = cp.zeros_like(img_gpu)
    grad_y = cp.zeros_like(img_gpu)
    
    grad_x[:, :-1] = img_gpu[:, 1:] - img_gpu[:, :-1]
    grad_y[:-1, :] = img_gpu[1:, :] - img_gpu[:-1, :]
    
    return grad_x, grad_y

def _compute_laplacian_field_gpu(grad_x, grad_y):
    div = cp.zeros_like(grad_x)
    

    div[:, 0] = grad_x[:, 0]
    div[:, 1:] = grad_x[:, 1:] - grad_x[:, :-1]
    

    div[0, :] += grad_y[0, :]
    div[1:, :] += grad_y[1:, :] - grad_y[:-1, :]
    
    return div

# ---------------------------
# Reconstruct from gradients (GPU)
# ---------------------------
def reconstruct_from_gradients(gradx, grady, boundary_img):
    gradx_g = cp.asarray(gradx, dtype=cp.float64)
    grady_g = cp.asarray(grady, dtype=cp.float64)
    bound_g = cp.asarray(boundary_img, dtype=cp.float64)
    
    h, w = gradx_g.shape
    
    interior_mask = cp.ones((h, w), dtype=bool)
    interior_mask[0, :] = False
    interior_mask[-1, :] = False
    interior_mask[:, 0] = False
    interior_mask[:, -1] = False
    
    n_pixels = int(cp.sum(interior_mask))
    
    A, idx_map = _build_laplacian_matrix_gpu(n_pixels, interior_mask)
    
    div = _compute_laplacian_field_gpu(gradx_g, grady_g)
    
    b = -div[interior_mask]
    
    coords = cp.argwhere(interior_mask)
    y_in, x_in = coords[:, 0], coords[:, 1]
    
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    lin_indices = cp.arange(n_pixels, dtype=cp.int32)
    
    for dy, dx in neighbors:
        ny, nx = y_in + dy, x_in + dx
        
        is_boundary = ~interior_mask[ny, nx]
        
        if cp.any(is_boundary):
            idx_b = lin_indices[is_boundary]
            val_b = bound_g[ny[is_boundary], nx[is_boundary]]
            
            cupyx.scatter_add(b, idx_b, val_b)
            
    # Solve
    x_sol, _ = cg(A, b, tol=1e-5)
    
    res = bound_g.copy()
    res[interior_mask] = x_sol
    
    return res.get()

# ---------------------------
# Seamless Cloning (GPU)
# ---------------------------
def seamless_clone(src, dst, mask, offset, mixed=True, tol=1e-5):
    src_g = cp.asarray(src, dtype=cp.float64)
    dst_g = cp.asarray(dst, dtype=cp.float64)
    mask_g = cp.asarray(mask) > 0
    
    h_s, w_s = mask_g.shape
    y_off, x_off = offset
    h_d, w_d, _ = dst_g.shape
    
    mask_dst = cp.zeros((h_d, w_d), dtype=bool)
    if y_off < 0 or x_off < 0 or (y_off + h_s) > h_d or (x_off + w_s) > w_d:
        raise ValueError("Source offset out of bounds.")
    
    mask_dst[y_off:y_off+h_s, x_off:x_off+w_s] = mask_g
    
    n_pixels = int(cp.sum(mask_dst))
    if n_pixels == 0:
        return dst_g.get()
        
    A, idx_map = _build_laplacian_matrix_gpu(n_pixels, mask_dst)
    
    coords_in = cp.argwhere(mask_dst)
    y_in, x_in = coords_in[:, 0], coords_in[:, 1]
    
    y_src = y_in - y_off
    x_src = x_in - x_off
    
    out_g = dst_g.copy()

    for ch in range(3):
        src_ch = src_g[:, :, ch]
        dst_ch = dst_g[:, :, ch]
        
        # Gradients
        gs_x, gs_y = _compute_gradients_gpu(src_ch)
        
        if mixed:
            gd_x, gd_y = _compute_gradients_gpu(dst_ch)
            
            val_gs_x = gs_x[y_src, x_src]
            val_gs_y = gs_y[y_src, x_src]
            val_gd_x = gd_x[y_in, x_in]
            val_gd_y = gd_y[y_in, x_in]
            
            mag_s = val_gs_x**2 + val_gs_y**2
            mag_d = val_gd_x**2 + val_gd_y**2
            
            use_dst = mag_d > mag_s
            final_gx_vals = cp.where(use_dst, val_gd_x, val_gs_x)
            final_gy_vals = cp.where(use_dst, val_gd_y, val_gs_y)
        else:
            final_gx_vals = gs_x[y_src, x_src]
            final_gy_vals = gs_y[y_src, x_src]
            
        # Reconstruct temp gradient field to compute Laplacian
        temp_gx = cp.zeros_like(src_ch)
        temp_gy = cp.zeros_like(src_ch)
        
        temp_gx[y_src, x_src] = final_gx_vals
        temp_gy[y_src, x_src] = final_gy_vals
        
        lap_field = _compute_laplacian_field_gpu(temp_gx, temp_gy)
        
        b = -lap_field[y_src, x_src]
        
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        linear_indices = cp.arange(n_pixels, dtype=cp.int32)
        
        for dy, dx in neighbors:
            ny, nx = y_in + dy, x_in + dx
            
            valid = (ny >= 0) & (ny < h_d) & (nx >= 0) & (nx < w_d)
            ny_v = ny[valid]
            nx_v = nx[valid]
            
            is_boundary = ~mask_dst[ny_v, nx_v]
            
            ny_b = ny_v[is_boundary]
            nx_b = nx_v[is_boundary]
            
            lin_idx_v = linear_indices[valid]
            lin_idx_b = lin_idx_v[is_boundary]
            
            boundary_vals = dst_ch[ny_b, nx_b]
            cupyx.scatter_add(b, lin_idx_b, boundary_vals)

        x_sol, info = cg(A, b, tol=tol, maxiter=2000)
        out_g[y_in, x_in, ch] = cp.clip(x_sol, 0, 255)

    return out_g.get()