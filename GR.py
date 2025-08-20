# === Phase-Lattice → N(u) → Kα(u) → Auto-Rescale Plateau (Dual-Alpha, v_transport print, unified tuning) ===
import numpy as np
from math import gamma as Gamma

# ======================= 0) 用户零点（可选） =======================
# 若你有自己的正半轴零点 γ_n，请粘到 gamma_user；否则使用内置前60个。
gamma_user = []  # e.g., [14.1347, 21.0220, ...]

# ======================= 1) 相位格子 & N(u) =======================
# 网格与相位映射参数（按建议做了稳健性调节）
u_min, u_max, num_points = 10.0, 200.0, 2000
sigma_phase = 0.5
b = 1.0
a = 0.34       # ↓ 轻微降低 a
tau0 = 18.0    # ↑ 提高 tau0
FORCE_MEAN_N_TO_1 = True

u = np.linspace(u_min, u_max, num_points)
du = u[1]-u[0]

# 内置前60个零点（正半轴）
gamma_pos_default = np.array([
    14.134725141734694,21.022039638771555,25.01085758014569,30.424876125859513,
    32.93506158773919,37.58617815882567,40.918719012147495,43.327073280915,
    48.00515088116716,49.7738324776723,52.97032147771446,56.446247697063396,
    59.34704400260235,60.83177852460981,65.11254404808161,67.07981052949417,
    69.54640171117398,72.06715767448191,75.70469069908393,77.1448400688748,
    79.33737502024937,82.91038085408603,84.73549298051705,87.42527461312523,
    88.80911120763447,92.49189927055848,94.65134404051989,95.87063422824531,
    98.83119421819369,101.31785100573139,103.72553804047834,105.4466230523261,
    107.16861118427641,111.02953554316967,111.87465917699268,114.32022091545272,
    116.22668032117026,118.7907828659763,121.37012500242065,122.94682929355259,
    124.25681855434577,127.51668387959699,129.57870419968777,131.08768853157843,
    133.4977372029976,134.7565097533739,138.11604205453344,139.73620895212138,
    141.12370740402193,143.11184580762063,146.00098248648615,147.422765343889,
    150.053520420784,150.925257612648,153.02469381138,156.112909294674,
    157.597591817096,158.84998817142,161.188964137596,163.030709687043
], dtype=float)

gamma_pos = np.array(gamma_user, dtype=float) if len(gamma_user)>0 else gamma_pos_default
gamma_signed = np.concatenate([+gamma_pos, -gamma_pos])  # 对称 ±γ

def gauss_smooth(y, sigma_pts=40):
    W = int(6*sigma_pts) | 1
    x = np.arange(W) - W//2
    g = np.exp(-(x**2)/(2*(sigma_pts**2))); g /= g.sum()
    yp = np.pad(y, (W//2, W//2), mode="reflect")
    return np.convolve(yp, g, mode="valid")

# 相位 → τ → N → v_eff → T_int
phi = np.arctan((u[:,None]-gamma_signed[None,:])/sigma_phase).sum(axis=1)
tau = b*(phi - np.nanmean(phi))
tau_s = gauss_smooth(tau, sigma_pts=40)
N = np.exp(a*np.tanh(tau_s/tau0))
if FORCE_MEAN_N_TO_1:
    N = N/np.nanmean(N)
c0 = 1.0
v_eff = c0/N
T_int = (1.0/c0)*np.cumsum(N)*du

# 按建议：打印传输平均速度，避免 <v_eff> 误解
v_transport = (u.max() - u.min()) / T_int[-1]
print("=== Phase-driven Index (Riemann lattice) ===")
print(f"<N>={float(np.mean(N)):.4f}, v_transport≈{float(v_transport):.4f} (c0=1), "
      f"T_int(u_max)≈{float(T_int[-1]):.4f} over Δu={u.max()-u.min():.1f}")

# ======================= 2) K_alpha(u) 计算（调参：smooth_log=40; base/max win 调大） =======================
def central_diff(y, dx):
    d = np.zeros_like(y)
    d[1:-1] = (y[2:] - y[:-2])/(2*dx)
    d[0]    = (y[1]-y[0])/dx
    d[-1]   = (y[-1]-y[-2])/dx
    return d

def caputo_D_alpha(y, dx, alpha):
    fp = central_diff(y, dx)
    n = len(y); m = np.arange(1, n+1, dtype=float)
    w = (m*dx)**(-alpha)
    z = np.convolve(fp, w, mode='full')[:n]
    return (dx / Gamma(1.0 - alpha)) * z

def local_linear_deriv(x, y, i, half):
    n=len(x); i0,i1=max(0,i-half),min(n-1,i+half)
    xs=x[i0:i1+1]; ys=y[i0:i1+1]; xc=xs-xs.mean()
    denom=np.dot(xc,xc)+1e-12
    return float(np.dot(xc,ys-ys.mean())/denom)

def adaptive_derivative(x, y, base_win=30, max_win=120, thr=8e-4):
    dx = x[1]-x[0]
    raw=central_diff(y, dx); out=np.zeros_like(y)
    for i in range(len(y)):
        half = base_win if abs(raw[i])>=thr else max_win
        out[i]=local_linear_deriv(x, y, i, half)
    return out

def build_K_alpha(u, nu_field, alpha, smooth_log=40, K_star=2.6667):
    Phi  = caputo_D_alpha(nu_field, u[1]-u[0], alpha)
    logP = np.log(np.maximum(1e-18, Phi))
    logPs= gauss_smooth(logP, sigma_pts=smooth_log)
    dlogP2  = adaptive_derivative(u, logP,  base_win=30, max_win=120, thr=8e-4)
    dlogHs2 = adaptive_derivative(u, logPs, base_win=30, max_win=120, thr=8e-4)
    dlogH2  = K_star * dlogHs2
    K_al    = np.divide(dlogH2, dlogP2, out=np.full_like(dlogH2, np.nan), where=np.abs(dlogP2)>=1e-18)
    return K_al

alphas = [0.5, 0.7]
K_curves = {alpha: build_K_alpha(u, tau_s, alpha, smooth_log=40, K_star=2.6667) for alpha in alphas}

# ======================= 3) 平台检测（自动调参 + 伪平台防御） =======================
def _smooth(K, sigma_pts=40):
    W = int(6*sigma_pts) | 1
    x = np.arange(W)-W//2
    g = np.exp(-(x**2)/(2*(sigma_pts**2))); g /= g.sum()
    return np.convolve(np.pad(K,(W//2,W//2),'reflect'), g, mode='valid')

def detect_plateaus(u, K, *, win=260, slope_tol=2e-3, std_tol=0.06,
                    target=None, val_tol=0.18, min_len=25.0,
                    max_len_frac=0.90, sigma_pts=40):
    du = u[1]-u[0]; half = win//2
    Ksm = _smooth(K, sigma_pts=sigma_pts)
    raw=[]
    for i in range(half, len(u)-half):
        xs = u[i-half:i+half+1]; ys = Ksm[i-half:i+half+1]
        xc = xs - xs.mean()
        slope = float(((ys-ys.mean())*xc).sum()/((xc*xc).sum() + 1e-12))
        std   = float(ys.std()); meanK = float(ys.mean())
        ok = (abs(slope) <= slope_tol) and (std <= std_tol)
        if target is not None: ok &= (abs(meanK - target) <= val_tol)
        if ok: raw.append((u[i-half], u[i+half], meanK, std))
    # 合并连续
    merged=[]
    if raw:
        cur=list(raw[0])
        for s in raw[1:]:
            if s[0] <= cur[1] + du: cur[1]=s[1]
            else: merged.append(tuple(cur)); cur=list(s)
        merged.append(tuple(cur))
    # 过滤短段/近全域
    U=u.max()-u.min()
    return [(a,b,m,s) for (a,b,m,s) in merged if (b-a)>=min_len and (b-a)<=max_len_frac*U]

def best_segment(u, segs):
    if not segs: return None
    u0,u1,km,ks = max(segs, key=lambda t:(t[1]-t[0]))
    length=u1-u0; cover=100.0*length/(u.max()-u.min())
    pts=int(((u>=u0)&(u<=u1)).sum())
    return dict(seg=(float(u0), float(u1)), Kmean=float(km), Kstd=float(ks),
                length=float(length), cover=float(cover), pts=int(pts))

def ci95(std, n): return 1.96*std/max(1, np.sqrt(max(1, n)))

# 自动调参日程（严格→放松）
schedule = [
    (40, 260, 2.0e-3, 0.06, 0.18, 25.0),
    (40, 240, 2.5e-3, 0.07, 0.20, 22.0),
    (40, 220, 3.0e-3, 0.08, 0.22, 20.0),
    (40, 200, 3.5e-3, 0.09, 0.24, 18.0),
    (40, 180, 4.0e-3, 0.10, 0.25, 16.0),
    (40, 160, 5.0e-3, 0.12, 0.28, 14.0),
]

def auto_find(u, K, K_star, schedule, max_len_frac=0.90):
    # baseline（无 target）：用于估计 rescale
    base=None
    for (sig,win,sl,st,vv,ml) in schedule[:3]:
        segs = detect_plateaus(u, K, win=win, slope_tol=sl, std_tol=st, target=None,
                               val_tol=None, min_len=ml, max_len_frac=max_len_frac, sigma_pts=sig)
        cand = best_segment(u, segs)
        if cand and ((base is None) or (cand['length']>base['length'])):
            base=cand
    scale = (K_star/base['Kmean']) if base else None
    K_use = K*scale if scale else K
    # 带 target 的命中
    for (sig,win,sl,st,vv,ml) in schedule:
        segs = detect_plateaus(u, K_use, win=win, slope_tol=sl, std_tol=st, target=K_star,
                               val_tol=vv, min_len=ml, max_len_frac=max_len_frac, sigma_pts=sig)
        hit = best_segment(u, segs)
        if hit:
            return dict(hit=hit, params=dict(sigma=sig, win=win, slope=sl, std=st, val=vv, min_len=ml),
                        scale=scale, base=base)
    return dict(hit=None, params=None, scale=scale, base=base)

def auto_find_with_fixed_scale(u, K, K_star, schedule, fixed_scale=None, max_len_frac=0.90):
    # 使用固定 scale（跨 α 一致性检验）
    base=None
    if fixed_scale is None:
        return auto_find(u, K, K_star, schedule, max_len_frac)
    K_use = K*fixed_scale
    for (sig,win,sl,st,vv,ml) in schedule:
        segs = detect_plateaus(u, K_use, win=win, slope_tol=sl, std_tol=st, target=K_star,
                               val_tol=vv, min_len=ml, max_len_frac=max_len_frac, sigma_pts=sig)
        hit = best_segment(u, segs)
        if hit:
            return dict(hit=hit, params=dict(sigma=sig, win=win, slope=sl, std=st, val=vv, min_len=ml),
                        scale=fixed_scale, base=None)
    return dict(hit=None, params=None, scale=fixed_scale, base=None)

# 运行 α=0.5 / 0.7（自适应重标定）
K_star = 2.6667
res = {}
for alpha in alphas:
    res[alpha] = auto_find(u, K_curves[alpha], K_star, schedule)

# 跨 α 一致性：用 α=0.5 的 scale 去校正 α=0.7
scale_05 = res[0.5]['scale']
res_07_consis = auto_find_with_fixed_scale(u, K_curves[0.7], K_star, schedule, fixed_scale=scale_05)

# 重叠（两 α 均命中才算）
overlap=None
if res[0.5]['hit'] and res[0.7]['hit']:
    a0,a1 = res[0.5]['hit']['seg']; b0,b1 = res[0.7]['hit']['seg']
    o0,o1 = max(a0,b0), min(a1,b1)
    if o1>o0:
        overlap = dict(seg=(o0,o1), length=o1-o0,
                       cover=100.0*(o1-o0)/(u.max()-u.min()))

# ======================= 4) 打印总结（简洁） =======================
def line(alpha, R):
    tag=f"[α={alpha} @ K*={K_star}]"
    if R['hit']:
        h, p = R['hit'], R['params']
        ci = ci95(h['Kstd'], h['pts']); s0,s1 = h['seg']
        sc = R['scale']; sc_txt = (f", scale={sc:.6f}" if sc else "")
        return (f"{tag} HIT  seg=({s0:.3f}, {s1:.3f})  len={h['length']:.3f}  "
                f"K̄={h['Kmean']:.4f} ± {ci:.4f}  σ={h['Kstd']:.4f}  "
                f"cover={h['cover']:.1f}%  pts={h['pts']}  "
                f"params={{σ:{p['sigma']}, win:{p['win']}, slope:{p['slope']:.1e}, std:{p['std']:.2f}, val:{p['val']}, min_len:{p['min_len']}}}"
                f"{sc_txt}")
    # NO-HIT → 输出 nearest baseline（若有）
    nb = R['base']
    if nb:
        ci = ci95(nb['Kstd'], nb['pts']); s0,s1 = nb['seg']
        sc = R['scale']; sc_txt = (f", scale={sc:.6f}" if sc else "")
        return (f"{tag} NO-HIT  (nearest baseline) seg=({s0:.3f}, {s1:.3f})  "
                f"len={nb['length']:.3f}  K̄={nb['Kmean']:.4f} ± {ci:.4f}  "
                f"σ={nb['Kstd']:.4f}  cover={nb['cover']:.1f}%{sc_txt}")
    return f"{tag} NO-HIT"

def line_consistency(tag, R):
    if R['hit']:
        h, p = R['hit'], R['params']
        ci = ci95(h['Kstd'], h['pts']); s0,s1 = h['seg']
        return (f"{tag} HIT  seg=({s0:.3f}, {s1:.3f})  len={h['length']:.3f}  "
                f"K̄={h['Kmean']:.4f} ± {ci:.4f}  σ={h['Kstd']:.4f}  "
                f"cover={h['cover']:.1f}%  pts={h['pts']}  "
                f"params={{σ:{p['sigma']}, win:{p['win']}, slope:{p['slope']:.1e}, std:{p['std']:.2f}, val:{p['val']}, min_len:{p['min_len']}}}  "
                f"[using α=0.5 scale]")
    return f"{tag} NO-HIT  [using α=0.5 scale]"

print("=== Phase-Lattice → N(u) → Kα(u) → Auto-Rescale Plateau ===")
print(line(0.5, res[0.5]))
print(line(0.7, res[0.7]))
if overlap:
    o0,o1 = overlap['seg']
    print(f"[Overlap 0.5∩0.7] seg=({o0:.3f}, {o1:.3f})  len={overlap['length']:.3f}  cover={overlap['cover']:.1f}%")
else:
    print("[Overlap 0.5∩0.7] none or single-hit only")
print(line_consistency("[Cross-α Consistency: α=0.7 with α=0.5-scale]", res_07_consis))
print("=== End ===")
