# MAP_ROW_SCHEMA.md — the row schema a map row must satisfy to be readable

Written this tick to close the deferral the 11:23Z reflection named (type-A band + measured type-B bound
into the schema). A row that cannot fill these fields is not a map row; it is a diagnostic.

## Required fields (hard-fail; enforced by lab/s109c_schema_check.py)

```
{
  "task":        {"level": int, "problem_id": int, "name": str},
  "op_class":    str,
  "procedure":   {"id": "A"|"B"|str, "cudnn_tf32": bool, "matmul_tf32": bool, "dtype": "fp32",
                  "torch_rev": str, "arch": "sm_120", "repo_rev": str, "commit": str},
  "reference":   {"engine_label": str, "top1_kernel": str, "label_basis": "profiler_top1_dominance"},
  "deviation":   {"statistic": "evaluator_exact_rho", "gate": float, "rho_ew": float, "frac": float,
                  "anchor": "chunked_fp64", "anchor_selfcheck_rel": float},
  "type_b":      {"quantity": "rho_ew", "procedure_pair": [str, str], "ratio": float, "source": str},
  "type_a":      {"n_draws": int, "cv": float, "min": float, "max": float, "exceed_rate": float,
                  "draw_axis": "task_instance_seed", "source": str},
  "verdict":     {"kind": "label"|"rate", "value": bool|float,
                  "rule": "label iff rho_ew>1 and exceed_rate==1 else rate"},
  "scope":       {"arch": str, "repo_rev": str, "procedure": str, "n_draws": int, "notes": str}
}
```

## Rules

1. `verdict.kind` is `rate` whenever `0 < exceed_rate < 1`, or whenever the row was measured on fewer than
   16 draws and sits inside rho [0.85, 1.30]. A single-draw row inside that band must be labelled
   `unmeasured_rate`, not pass/fail (s109b: L2/77 exceeds on 15/16 draws).
2. `type_a` is required for every row that ships a verdict. A row without a draw band may ship a
   *magnitude* statement only, marked `diagnostic`.
3. `type_b` is required whenever the row's procedure differs from the procedure its engine label was
   observed under. Measured bound: s106d 11-358x per-row for gemm between flag regimes, 1.000 for conv
   controls; s107 is the same bound applied to 191 rows.
4. `reference.label_basis` is always `profiler_top1_dominance`; a row whose top-1 is a non-conv kernel
   (s109: `at::native::elementwise_kernel` on 2/16 draws of L2/15) carries that name, and the label is
   read as "dominant kernel in the call", never as "the conv engine".
5. Provenance: `procedure.commit` must be a commit that exists in the local repo before the row's
   `measured_at`; every number carries its artifact path.

## Status

s109/s109b rows satisfy this schema (11 rows: 8 away-from-boundary labels, 1 boundary rate, 2 dgrad
magnitude rows). The remaining 180 rows of the 191-task map do NOT yet carry `type_a` and are therefore
*diagnostics* under rule 2 — declared, not hidden. Extending the draw band to the 23 rows in [0.5, 2] is
the next schema-closing measurement.
