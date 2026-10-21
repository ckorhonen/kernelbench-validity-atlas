"""D2 FREEZE -- pre-registration constants for the conv third-family test.

Run BEFORE the GPU harness; uses ONLY committed artifacts (guard_v1.json,
d1_cellaffine.json, b2_failure_corpus.jsonl) and the family declaration below.
No labels, no GPU, no measurements. Output: results/d2_freeze.json.

CONV THIRD-FAMILY DECLARATION (frozen 2026-10-20 ~21:00Z, before the run;
plan v2 item 3; conjecture rider per conjecture_offset_law.md):

Family: conv-as-gemm reassociation, b3 arm structure, ONE geometry.
  ref   = F.conv2d(x, W, b)                      (task-native conv, fp32)
  cand  = gemm on unfolded patches:
          P = x.unfold(2,3,1).unfold(3,3,1 reshaped (N,Oh,Ow,Cin,k,k)
              -> (N*Oh*Ow, Cin*9); T = P @ Wflat.T + b  (Wflat=(Cout,Cin*9))
          reshaped back to (N,Cout,Oh,Ow).
  Geometry (single, fixed): Cin=64, Cout=128, k=3, Hin=Win=8, stride 1,
    padding 0 (valid) => Oh=Ow=6.  x: randn(N,Cin,8,8), FIXED per seed.
  Arms (b3/b4 verbatim structure, conv instance):
    ordmag: 24 log-scales 1e0..1e6 (10**(6*si/23)), W=base*scale; b unscaled.
    cancel: scales {1e4,1e6} x 5 draws; W[0,0,0,0]=0 (structural zero, the
            b2 r_[0,0]=0 device on the first tap); x = one sample
            expand(N,Cin,8,8).contiguous() (identical batch rows, b2 device).
  Seeds {52,53} (fresh; corpus 0-4,43,44,45; b4 46-49; b5 50,51).
  label per b4 summarize(cand,ref): d=|T-R|, limit=ATOL+RTOL*|a|, ATOL=RTOL=1e-4.
  Features POST-scale W (b4 convention), on Wflat (the gemm's B operand -- the
  reassociation's multiplicative parameter; 'w_* on W verbatim').

ROLE MAPPING (the only family mapping; declared ex ante, b3 roles):
  B := N*Oh*Ow  (gemm rows / batch positions)         = 144
  K := Cin*k*k   (contracted extent)                   = 576
  H := Cout      (per-row output extent)               = 128
  collapse_work_ratio := B*K*H/(H*K+K)  (b2 formula, roles substituted)
  w_max/mean_abs/rel_spread over |Wflat| as b4.
  Constant-part within family: geometry fixed => {B,K,H,fan_in_sum,cwr} all
  family-constant; of the KEPT trio {w_mean_abs,w_max,cwr} only cwr is
  family-constant (w_* vary with the scale/draw arms) -- same structure as b5.

FROZEN LAW (deployment form; D1 commit 798ee44):
  T_u = a_dag + gamma * d_cen,u + b_F,  a_dag=-1.1052271376183762,
  gamma=+0.5876085080140739  (M1 beta, results/d1_cellaffine.json);
  d_cen,u = unit mean signed-z minus conv-family unit mean (label-free).
  RMSE_D1 (M1 residual SD, 25 units, recomputed here) = RMSE below.

CONJECTURE FREEZE (the item-3 rider; literal constant-part form):
  b_F^pred = ( z_cwr^True(v_cwr,F) - E_b4train[z_cwr^True] ) / 3
  (score = mean of 3 kept z's; a constant-part shift Delta-z moves the score
  Delta/3; the form assumes the family intercept tracks that shift 1:1).
  PASS iff |b_pred - b_meas| <= 2*RMSE_D1  ("M1-residual scale", 2-sigma
  reading, frozen; 1xRMSE also recorded). FAIL branch predeclared by the
  conjecture itself: free-offset reframe recorded, hierarchy result, lane
  continues per plan v2 kills -- NO additional lane consequence.
  b_meas := mean(t*_conv units) - a_dag   (d_cen centered per family).
  b5 BACKTEST of the same form, from committed artifacts only, declared
  ex ante as context (form chosen WITHOUT reference to any conv data):
  b5 pred -0.5741 vs b5 measured +0.8461 => residual +1.4202 = 10.2x RMSE_D1.
  The form is known-mislevel on b5; the conv test runs anyway as the
  conjecture's own registered serious attempt. Full 5-form table in JSON.

KILLS / SIGNALS (frozen; plan v2 item 3 verbatim numbers):
  K-a frozen-t sufficient: pooled conv precision@t_frozen(-0.6134445630270532)
      >= 0.90 AND accept >= 0.20 => no law needed (note only).
  K-b repaired precision < 0.80: law-T pooled precision < 0.80 => hierarchy
      wrong; 'per-family refit unavoidable' becomes the result.
  K-c cell-fit R^2 < 0.5: pooled 3-family M1+conv_ind fit R^2 < 0.5 =>
      decomposition empty.
  G-signal (construction-facing): law-T precision >= 0.90 AND accept >= 0.20
      on conv (a family with ZERO fit points in a_dag/gamma).
  Band 0.80-0.90 law precision: recorded intermediate, no kill, no signal.
  Youden t* per conv unit: d1 convention verbatim (plateau midpoint);
  units lacking both classes excluded per d1 rule and counted; if BOTH arms
  single-class => declared contingency: ordmag widened to 48 log steps
  1e0..1e8, same seeds, ONE rerun. AUC per arm pooled: descriptive.

All numbers below computed, printed, hashed into results/d2_freeze.json.
"""
import json, hashlib, statistics as st, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

def zof_deploy(v, col):
    # b4 deployed_scores verbatim: single-sided rank of appended value
    arr = sorted(list(col) + [v])
    n = len(arr)
    rk = arr.index(v) + 1  # value appended once; ties -> first slot == b4 rankdata[-1]? see note
    # b4: rankdata(cols+[v])[-1] == position of v in sorted(arr), ties->last? rankdata
    # in b4 assigns midrank to ties WITHIN the list; the appended value is unique
    # by construction only if not in col. Use exact b4 rankdata on col+[v].
    idx = sorted(range(n), key=lambda i: arr[i])
    out = [0.0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[idx[j + 1]] == arr[idx[i]]: j += 1
        m = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1): out[idx[k]] = m
        i = j + 1
    return out[idx.index(n - 1)] if False else out[list(range(n)).index(len(arr)-1) if False else 0]

def rank_last(v, col):
    """exact b4: rank of the LAST element of col+[v] under b4 rankdata midrank."""
    arr = col + [v]; n = len(arr)
    idx = sorted(range(n), key=lambda i: arr[i])
    out = [0.0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[idx[j + 1]] == arr[idx[i]]: j += 1
        m = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1): out[idx[k]] = m
        i = j + 1
    return out[n - 1]

def main():
    g = json.load(open(os.path.join(RES, "guard_v1.json")))
    ori = g["orientation"]; cols = g["col"] if "col" in g else g["corpus_columns"]
    cols = {k: list(v) for k, v in cols.items()}
    kept = g["kept_features"]; t_frozen = g["threshold"]
    d1 = json.load(open(os.path.join(RES, "d1_cellaffine.json")))
    beta = d1["M1_primary"]["beta"]; a_dag, gamma = beta[0], beta[2]
    tab = d1["table"]
    res = []
    for u in tab:
        mu = 1.0 if u["src"] == "b5" else 0.0
        res.append(u["t_star"] - (a_dag + beta[1] * mu + gamma * u["d_cen"]))
    rmse = (sum(r * r for r in res) / len(res)) ** 0.5

    # ---- geometry + cwr
    Cin, Cout, k, Hin, N = 64, 128, 3, 8, 4
    Oh = Ow = Hin - k + 1
    B_role, K_role, H_role = N * Oh * Ow, Cin * k * k, Cout
    cwr_conv = B_role * K_role * H_role / (H_role * K_role + K_role)
    z_cwr_conv = ori["collapse_work_ratio"] * (rank_last(cwr_conv, cols["collapse_work_ratio"]) - (len(cols["collapse_work_ratio"]) + 1) / 2) / (((len(cols["collapse_work_ratio"]) + 1) ** 2 - 1) / 12) ** 0.5

    # b4train expectation of the cwr z (committed rows, deployed convention)
    b2 = [json.loads(l) for l in open(os.path.join(RES, "b2_failure_corpus.jsonl"))]
    tr = [r for r in b2 if r.get("method") == "fold32"]
    assert len(tr) == 175
    n = len(cols["collapse_work_ratio"]) + 1
    sd = ((n * n - 1) / 12.0) ** 0.5; mu = (n + 1) / 2
    def zcwr(v):
        return ori["collapse_work_ratio"] * (rank_last(v, cols["collapse_work_ratio"]) - mu) / sd
    e_tr = st.fmean(zcwr(r["collapse_work_ratio"]) for r in tr)

    b_pred = (z_cwr_conv - e_tr) / 3.0
    bound2 = 2 * rmse

    # b5 backtest table (committed data only)
    b5 = [json.loads(l) for l in open(os.path.join(RES, "b5_stage2.jsonl"))]
    b4h = [json.loads(l) for l in open(os.path.join(RES, "b4_guard_v1.jsonl"))]
    Es_b5 = st.fmean(r["score"] for r in b5)
    Es_b4h = st.fmean(r["score"] for r in b4h)
    z16 = zcwr(16.0)
    forms = {
        "F1_const_div3": (z16 - e_tr) / 3,
        "F2_fullshift_vs_train": Es_b5 - 0.0,
        "F3_(1-g)*full": (1 - gamma) * Es_b5,
        "F5_g*full": gamma * Es_b5,
        "F6_fullshift_vs_held": Es_b5 - Es_b4h,
    }
    b5_meas = beta[1]
    backtest = {kk: {"pred": vv, "resid_meas_minus_pred": b5_meas - vv,
                     "over_x_rmse": (b5_meas - vv) / rmse} for kk, vv in forms.items()}

    out = dict(
        when__frozen="2026-10-20T21:00Z (before GPU run); see module docstring",
        a_dag=a_dag, gamma=gamma, rmse_d1=rmse, bound_2rmse=bound2,
        geometry=dict(Cin=Cin, Cout=Cout, k=k, Hin=Hin, N=N, Oh=Oh, Ow=Ow),
        roles=dict(B=B_role, K=K_role, H=H_role),
        cwr_conv=cwr_conv, z_cwr_conv_signed=z_cwr_conv, e_train_cwr_z=e_tr,
        b_conv_pred_F1=b_pred,
        b5_backtest=backtest, b5_meas=beta[1],
        t_frozen=t_frozen, kept=kept, ori=ori,
        corpus_sha=hashlib.sha256(open(os.path.join(RES, "b2_failure_corpus.jsonl"), "rb").read()).hexdigest(),
        seeds=[52, 53],
    )
    assert out["corpus_sha"] == g["corpus_sha256"], "corpus drift"
    json.dump(out, open(os.path.join(RES, "d2_freeze.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "b5_backtest"}, indent=1, default=str))
    print("b5 backtest:", json.dumps(backtest, indent=1))
    print("\nFROZEN: b_conv_pred=%.6f  pass iff |b_pred-b_meas|<=%.6f (2xRMSE_D1)" % (b_pred, bound2))

if __name__ == "__main__":
    main()
