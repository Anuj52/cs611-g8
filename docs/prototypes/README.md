Throwaway exploration scripts (not the submission code). Run from a clone of this repo so they can import
`formats.py` / `predictor.py`; data: `assignment_data.npz` from the course zip (or set `CS611_DATA` for the first four).

- proto_baseline.py, proto_analysis.py, proto_wa_bfp.py, proto_e4m1_bfp.py: first prototype of all formats (numpy only, standalone)
- stress_predictor.py: stress test of predictor.py (real datasets A-E, sparse W, structured X)
- phase_diagram_rule.py: feasibility test of the one-shot rule from a few statistics (plan section 10)
- review_q3/: independent reproduction of the Q3 numbers (see docs/REVIEW_Q3_REPORT.md)
