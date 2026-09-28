FAS Stage5D Cross-Scale Anchor Diagnostic v1

Purpose
-------
Stage5 v1.2 completed but C3 is not supported: FAS residual remains ~0.95+ and all three seeds abstain/Bank-Insufficient. Per CURRENT_DECISION_POINT.md, do NOT enter L7 blindly. This package performs a no-retraining diagnostic using preserved Stage3/4/5 artifacts.

It also checks a method-consistency issue: Stage5 legacy evaluation omitted the v0.11 global ancestor-norm normalization. That omission can change pi/support values, but cannot change the NNLS cone residual; therefore it cannot by itself rescue C3.

What this package tests
-----------------------
1) Recompute protocol-normalized FAS with the correct paired 1.7B base anchor.
2) Compare no-anchor, wrong-4B-anchor, and shuffled-1.7B-anchor controls.
3) Quantify held-out 4B-base -> 1.7B-base IFRF alignment.
4) Test direction-only normalization as a diagnostic.
5) Test anchor-only diagonal and linear-ridge transfer. Transfer fitting excludes the evaluated B32 probes; ridge strength is selected only by CV on remaining paired base anchors. No KD target response, support label, or exposure weight is used to fit the transfer.
6) Retain Absolute baseline context.

Run
---
cd ${FAS_WORK_ROOT}
tar -xzf FAS_Server_Stage5D_CrossScaleAnchorDiagnostic_v1.tar.gz
cd FAS_Server_Stage5D_CrossScaleAnchorDiagnostic_v1
chmod +x run_fas_server.sh
./run_fas_server.sh

Expected return file
--------------------
${FAS_WORK_ROOT}/FAS_SERVER_RETURN_STAGE5D_CROSSSCALE_ANCHOR_DIAGNOSTIC_V1.tar.gz

Upload that single return archive to Google Drive FAS folder, then tell ChatGPT only: 已上传

Scientific scope
----------------
This is a post-Stage5 diagnostic, not a confirmatory claim test. No transformed variant inherits the matched-G0 finite-sample guarantee. Construction/exposure weights remain diagnostic metadata, not functional ground truth.
