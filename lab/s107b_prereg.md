# s107b PREREG — declared extension of s107: re-key the REMAINING 107 map rows under procedure B

Frozen before any s107b row is computed. This is a declared extension of s107 (whose prereg covered the
85 flagged rows); it exists because s107 found gemm exceedances at 1e-4 rise from 1 (procedure A) to 28
(procedure B), so the map's totals are procedure-dependent for every row whose reference dispatches a
cublas matmul — including rows whose op_class label is not "gemm" (s71 labels by task name/op, not by
arithmetic actually run; L1/10 "3D tensor matrix multiplication" is labelled elementwise).

## Rows

The 107 remaining s105c rows with status ok and s71 anchor admissible, i.e. everything not in s107's
gemm|TF32 (46), convT|TF32 (31), conv|OTHER (8) selection. Ordered by s71 t_fp64_s ascending.

## Procedure

B only (cudnn tf32 True, matmul tf32 True), fp32, torch.manual_seed(0), anchor = float64 both flags off.
No procedure-A leg: A is already recorded by s71/s105c and s107 reproduced it on the flagged columns.
K1: procedure-B repeat bit-identical on the first 3 successful rows. K2: anchor repeat rel <= 1e-9 on the
first successful row. Instrument kill: n_ok < 90 => INSTRUMENT_FAILED.

## Measured / reported

Per row: label_B, top1_B, rho_ew and frac at 1e-4 and 1e-2, ref_max_abs, u_strict, wall_s. Summary
reports per-column medians and exceed counts under one procedure, and the repaired whole-map totals
(200 rows: s107 85 + s107b 107) at both gates, with the procedure-A totals from s105c side by side.
Predeclared reading: rows whose label_B is TF32_TENSOROP but whose s105c label was not are flagged as
"arithmetic not described by the op-class label"; no threshold gates that count, it is descriptive.
