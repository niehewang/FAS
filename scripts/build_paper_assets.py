#!/usr/bin/env python3
"""Generate LaTeX result tables, plots, and conservative result narration.

Workflow:
1. Fill CSV files under results/templates/ (leave unavailable cells blank or N/A).
2. Run: python scripts/build_paper_assets.py
3. Recompile main.tex.

The generator NEVER invents missing values. It narrates only mechanically
verifiable comparisons. Any scientific interpretation remains in the manuscript
and must be manually checked against statistics/failure cases before submission.
"""
from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "templates"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures" / "results"
NARR = ROOT / "results" / "generated_narrative.tex"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def esc(x):
    s = str(x)
    repl = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def show(x, digits=3):
    if pd.isna(x) or str(x).strip() == "":
        return r"\todoexp"
    if isinstance(x, str) and x.upper() == "N/A":
        return "N/A"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return esc(x)


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def read_csv(name):
    return pd.read_csv(RES / name, keep_default_na=False)


def write_main_table():
    df = read_csv("main_results.csv")
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{多父功能谱系恢复主结果。报告 end-to-end Parent-F1；对预先声明为 bank-complete 且 signal-bearing 的测试后代，选择性拒绝计为支持恢复失败。最后一列按各方法原生访问方式报告审计成本：FAS/输出基线为 API 调用，DNA-Decomp 为其固定 prompt 样本数，modelDNA 为权重访问；这些成本不强行换算为同一单位。}",
        r"\label{tab:main_results}", r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"方法 & Clean Merge & Merge+SFT & Multi-Teacher KD & Deep Chain & Audit access\\",
        r" & E2E-F1 & E2E-F1 & E2E-F1 & E2E-F1 & API/weights\\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        name = r"\textbf{FAS}" if str(r["method"]).strip().lower() == "fas" else esc(r["method"])
        lines.append(
            f"{name} & {show(r['clean_f1'])} & {show(r['sft_f1'])} & {show(r['kd_f1'])} & {show(r['deep_f1'])} & {show(r['audit_access'])}" + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES / "table_main_results.tex").write_text("\n".join(lines), encoding="utf-8")
    return df


def generic_table(filename, outname, caption, label, columns, headers, wide=False):
    df = read_csv(filename)
    fmt = "l" + "c" * (len(columns) - 1)
    env = "table*" if wide else "table"
    lines = [
        f"\\begin{{{env}}}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%" if wide else r"\resizebox{\columnwidth}{!}{%",
        f"\\begin{{tabular}}{{{fmt}}}", r"\toprule", " & ".join(headers) + r"\\", r"\midrule",
    ]
    for _, r in df.iterrows():
        vals = []
        for j, c in enumerate(columns):
            vals.append(esc(r[c]) if j == 0 else show(r[c]))
        lines.append(" & ".join(vals) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", f"\\end{{{env}}}"]
    (TABLES / outname).write_text("\n".join(lines), encoding="utf-8")
    return df


def build_tables():
    main = write_main_table()
    scaling = generic_table(
        "scaling.csv", "table_scaling.tex", "父模型数量与候选库规模扩展。", "tab:scaling",
        ["setting", "true_parents", "candidate_bank", "parent_f1", "exact_support", "l1_error"],
        ["Setting", "$s$", "$K$", "F1", "Exact", "$L_1$"],
    )
    validity = generic_table(
        "functional_validity.csv", "table_functional_validity.tex", "FAS 坐标与反事实功能贡献的一致性。", "tab:functional_validity",
        ["quantity", "spearman"], ["可解释量", "Spearman $\\rho$"],
    )
    openset = generic_table(
        "open_set.csv", "table_open_set.tex", "开放集祖先与 conformal calibration。", "tab:open_set",
        ["setting", "auroc", "fpr95", "false_abstain_delta05"],
        ["Setting", "AUROC", "FPR@95", r"False abstain ($\delta=.05$)"],
    )
    generic_table(
        "crossscale.csv", "table_crossscale.tex", "跨尺度蒸馏谱系恢复与 anchor 消融。", "tab:crossscale",
        ["setting", "anchor", "parent_f1", "decomposable_rate", "queries"], ["Setting", "Anchor", "F1", "Decomp. rate", "Queries"],
    )
    generic_table(
        "crossmodality.csv", "table_crossmodality.tex", "跨模态验证。", "tab:crossmodality",
        ["modality", "setting", "parent_f1", "decomposable_rate", "queries"], ["Modality", "Setting", "F1", "Decomp. rate", "Queries"],
    )
    generic_table(
        "ablation.csv", "table_ablation.tex", "主要消融实验。", "tab:ablation",
        ["variant", "parent_f1", "l1_error", "open_auroc"], ["Variant", "F1", "$L_1$", "Open AUROC"],
    )
    sampling = generic_table(
        "sampling_budget.csv", "table_sampling_budget.tex", "有限采样与在线查询预算。", "tab:sampling_budget",
        ["m", "strategy", "sigma_min_raw", "l1_error", "online_calls"],
        ["$m$", "Strategy", "$\\sigma_{\\min}^{\\mathrm{raw}}$", "$L_1$", "Online calls"],
    )
    geometry = generic_table(
        "geometry_mechanism.csv", "table_geometry_mechanism.tex", "字典几何与恢复误差之间的机制相关性。", "tab:geometry_mechanism",
        ["quantity", "spearman", "ci_low", "ci_high"], ["Quantity", "Spearman $\\rho$", "CI low", "CI high"],
    )
    encoder = generic_table(
        "encoder_sensitivity.csv", "table_encoder_sensitivity.tex", "输出表示器敏感性。", "tab:encoder_sensitivity",
        ["encoder", "parent_f1", "coordinate_spearman", "support_jaccard"],
        ["Encoder", "F1", "Coord. $\\rho$", "Support Jaccard"],
    )
    generic_table(
        "selective_coverage.csv", "table_selective_coverage.tex", "选择性谱系推断的覆盖率与条件恢复质量。", "tab:selective_coverage",
        ["setting", "decomposable_rate", "low_signal_rate", "bank_insufficient_rate", "conditional_parent_f1"],
        ["Setting", "Decomp. rate", "Low-signal", "Bank-insuff.", "Cond. F1"], wide=True,
    )
    generic_table(
        "negative_controls.csv", "table_negative_controls.tex", "Low-Signal gate 与负对照。", "tab:negative_controls",
        ["control", "false_positive_or_f1", "signal_pass_or_expected", "interpretation"],
        ["Control", "False-pos./F1", "Pass/Expected", "Interpretation"], wide=True,
    )
    generic_table(
        "coordinate_uncertainty.csv", "table_coordinate_uncertainty.tex", "代表性后代的 FAS 坐标经验不确定性。", "tab:coordinate_uncertainty",
        ["setting", "ancestor", "coordinate", "ci_low", "ci_high", "support_frequency"],
        ["Setting", "Ancestor", "Coord.", "CI low", "CI high", "Support freq."], wide=True,
    )
    return main, scaling, validity, openset, sampling, geometry, encoder


def plot_active():
    df = read_csv("active_probe_curve.csv")
    for c in ["budget", "sigma_min_raw", "sigma_min_unit", "coherence", "l1_error"]:
        df[c] = numeric(df[c])
    if df["l1_error"].notna().sum() < 2:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for method, g in df.groupby("method"):
        gg = g.dropna(subset=["budget", "l1_error"]).sort_values("budget")
        if len(gg):
            ax.plot(gg["budget"], gg["l1_error"], marker="o", label=method)
    ax.set_xlabel("Probe budget B")
    ax.set_ylabel("Coordinate L1 error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "active_probe_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_grouped(csv_name, xcol, groupcol, ycol, outfile, ylabel):
    df = read_csv(csv_name)
    df[ycol] = numeric(df[ycol])
    if df[ycol].notna().sum() < 2:
        return
    piv = df.pivot(index=xcol, columns=groupcol, values=ycol)
    ax = piv.plot(kind="bar", figsize=(5.2, 3.2))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(FIGS / outfile, bbox_inches="tight")
    plt.close()


def plot_probe_types():
    df = read_csv("probe_type_stats.csv")
    df["selected_fraction"] = numeric(df["selected_fraction"])
    if df["selected_fraction"].notna().sum() < 2:
        return
    g = df.dropna(subset=["selected_fraction"])
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(g["intervention_type"], g["selected_fraction"])
    ax.set_ylabel("Selected fraction")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGS / "probe_type_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def find_row(df, name):
    m = df[df.iloc[:, 0].astype(str).str.lower() == name.lower()]
    return None if m.empty else m.iloc[0]


def active_summary():
    df = read_csv("active_probe_curve.csv")
    for c in ["budget", "sigma_min_raw", "sigma_min_unit", "coherence", "l1_error"]:
        df[c] = numeric(df[c])
    a = df[df["method"].astype(str).str.lower() == "active"].dropna(subset=["budget", "l1_error"])
    r = df[df["method"].astype(str).str.lower() == "random"].dropna(subset=["budget", "l1_error"])
    if a.empty or r.empty:
        return r"\todoresult{填写 active\_probe\_curve.csv 后自动生成主动查询机制描述。}"
    common = sorted(set(a["budget"].astype(int)) & set(r["budget"].astype(int)))
    if not common:
        return r"\todoresult{Active/Random 缺少共同 query budget。}"
    b = common[-1]
    aa = a[a["budget"] == b].iloc[0]
    rr = r[r["budget"] == b].iloc[0]
    parts = [f"在最大共同 budget $B={b}$ 时，Active probes 的 coordinate $L_1$ error 为 {aa['l1_error']:.3f}，Random 为 {rr['l1_error']:.3f}"]
    if pd.notna(aa.get("sigma_min_raw")) and pd.notna(rr.get("sigma_min_raw")):
        parts.append(f"对应原始字典 $\\sigma_{{\\min}}^{{\\mathrm{{raw}}}}$ 分别为 {aa['sigma_min_raw']:.3f} 与 {rr['sigma_min_raw']:.3f}")
    if pd.notna(aa.get("sigma_min_unit")) and pd.notna(rr.get("sigma_min_unit")):
        parts.append(f"单位列角度诊断 $\\sigma_{{\\min}}^{{\\mathrm{{unit}}}}$ 分别为 {aa['sigma_min_unit']:.3f} 与 {rr['sigma_min_unit']:.3f}")
    return "；".join(parts) + "。"


def best_numeric_summary(main, validity, openset, scaling, sampling, geometry, encoder):
    fas = find_row(main, "FAS")
    rnd = find_row(main, "Random-IFRF + NNLS")
    main_txt = r"\todoresult{填写 main\_results.csv 后自动生成主结果描述。}"
    best_result = None
    if fas is not None and rnd is not None:
        vals = []
        for c in ["clean_f1", "sft_f1", "kd_f1", "deep_f1"]:
            a = pd.to_numeric(pd.Series([fas[c]]), errors="coerce").iloc[0]
            b = pd.to_numeric(pd.Series([rnd[c]]), errors="coerce").iloc[0]
            if pd.notna(a) and pd.notna(b):
                vals.append((c, float(a - b), float(a)))
        if vals:
            key, diff, val = max(vals, key=lambda t: t[1])
            label = {"clean_f1": "Clean Merge", "sft_f1": "Merge+SFT", "kd_f1": "Multi-Teacher KD", "deep_f1": "Deep Chain"}[key]
            main_txt = f"与 Random-IFRF 相比，FAS 在 {label} 上的 Parent-F1 提升 {diff:.3f}（达到 {val:.3f}）；其余设置见表~\\ref{{tab:main_results}}。"
            best_result = (label, val, diff)

    validity_txt = r"\todoresult{填写 functional\_validity.csv 后自动生成功能有效性描述。}"
    vf = validity[validity["quantity"].astype(str).str.lower() == "fas coordinates"]
    if not vf.empty:
        x = pd.to_numeric(vf["spearman"], errors="coerce").iloc[0]
        if pd.notna(x):
            validity_txt = f"FAS 坐标与反事实功能贡献的 Spearman 相关为 {float(x):.3f}（表~\\ref{{tab:functional_validity}}）。"

    open_txt = r"\todoresult{填写 open\_set.csv 后自动生成开放集描述。}"
    nums = numeric(openset["auroc"])
    if nums.notna().any():
        open_txt = f"开放集实验的可用 AUROC 范围为 {nums.min():.3f}--{nums.max():.3f}；同时报告 $\\delta=0.05$ 下的错误拒绝率以检验 calibration。"

    scale_txt = r"\todoresult{填写 scaling.csv 后自动生成父模型数量扩展描述。}"
    ss = scaling[scaling["setting"].astype(str).str.lower() == "parent scaling"].copy()
    ss["parent_f1"] = numeric(ss["parent_f1"])
    if ss["parent_f1"].notna().sum() >= 2:
        ss = ss.dropna(subset=["parent_f1"]).sort_values("true_parents")
        scale_txt = f"当真实父节点从 {int(ss.iloc[0]['true_parents'])} 增至 {int(ss.iloc[-1]['true_parents'])} 时，Parent-F1 从 {ss.iloc[0]['parent_f1']:.3f} 变化到 {ss.iloc[-1]['parent_f1']:.3f}。"

    sampling_txt = r"\todoresult{填写 sampling\_budget.csv 后自动生成有限采样描述。}"
    sm = sampling.copy(); sm["l1_error"] = numeric(sm["l1_error"]); sm["online_calls"] = numeric(sm["online_calls"])
    aa = sm[sm["strategy"].astype(str).str.lower() == "active"].dropna(subset=["l1_error"])
    if len(aa) >= 2:
        aa = aa.sort_values("m")
        sampling_txt = f"Active probes 下，将每个 query 的重复采样从 $m={int(aa.iloc[0]['m'])}$ 增至 $m={int(aa.iloc[-1]['m'])}$ 时，coordinate $L_1$ error 从 {aa.iloc[0]['l1_error']:.3f} 变化到 {aa.iloc[-1]['l1_error']:.3f}。"

    mechanism_txt = r"\todoresult{填写 geometry\_mechanism.csv 后自动生成机制相关性描述。}"
    gm = geometry[geometry["quantity"].astype(str).str.lower() == "sigma_min_raw_vs_l1"]
    if not gm.empty:
        x = numeric(gm["spearman"]).iloc[0]
        if pd.notna(x):
            mechanism_txt = f"跨 seed/budget 的原始字典 $\\sigma_{{\\min}}^{{\\mathrm{{raw}}}}$ 与 coordinate error 的 Spearman 相关为 {float(x):.3f}（表~\\ref{{tab:geometry_mechanism}}）。"

    encoder_txt = r"\todoresult{填写 encoder\_sensitivity.csv 后自动生成表示器敏感性描述。}"
    ef = encoder.copy(); ef["parent_f1"] = numeric(ef["parent_f1"])
    if ef["parent_f1"].notna().sum() >= 2:
        z = ef.dropna(subset=["parent_f1"])
        encoder_txt = f"不同输出表示器下 Parent-F1 位于 {z['parent_f1'].min():.3f}--{z['parent_f1'].max():.3f} 区间，详细一致性见表~\\ref{{tab:encoder_sensitivity}}。"

    act_txt = active_summary()

    abstract_tail = ""
    conclusion_txt = r"\todoresult{实验完成后自动生成结论中的核心数字。}"
    if best_result is not None:
        label, val, diff = best_result
        abstract_tail = f" 在 {label} 设置中，FAS 的 Parent-F1 达到 {val:.3f}，相对 Random-IFRF 提升 {diff:.3f}；完整统计与置信区间见正文。"
        conclusion_txt = f"在 {label} 上，FAS 的 Parent-F1 为 {val:.3f}，相较 Random-IFRF 提升 {diff:.3f}。"
    return main_txt, validity_txt, open_txt, scale_txt, sampling_txt, mechanism_txt, encoder_txt, act_txt, abstract_tail, conclusion_txt


def write_narrative(main, scaling, validity, openset, sampling, geometry, encoder):
    vals = best_numeric_summary(main, validity, openset, scaling, sampling, geometry, encoder)
    (main_txt, validity_txt, open_txt, scale_txt, sampling_txt, mechanism_txt,
     encoder_txt, active_txt, abstract_tail, conclusion_txt) = vals
    text = "\n".join([
        "% Auto-generated by scripts/build_paper_assets.py",
        f"\\providecommand{{\\ResultAbstractTail}}{{{abstract_tail}}}",
        f"\\providecommand{{\\ResultMainSummary}}{{{main_txt}}}",
        f"\\providecommand{{\\ResultActiveSummary}}{{{active_txt}}}",
        f"\\providecommand{{\\ResultValiditySummary}}{{{validity_txt}}}",
        f"\\providecommand{{\\ResultOpenSummary}}{{{open_txt}}}",
        f"\\providecommand{{\\ResultScaleSummary}}{{{scale_txt}}}",
        f"\\providecommand{{\\ResultSamplingSummary}}{{{sampling_txt}}}",
        f"\\providecommand{{\\ResultMechanismSummary}}{{{mechanism_txt}}}",
        f"\\providecommand{{\\ResultEncoderSummary}}{{{encoder_txt}}}",
        f"\\providecommand{{\\ResultConclusionSummary}}{{{conclusion_txt}}}",
    ])
    NARR.write_text(text, encoding="utf-8")


def main():
    main_df, scaling, validity, openset, sampling, geometry, encoder = build_tables()
    plot_active()
    plot_grouped("ancestry_trajectory.csv", "stage", "ancestor", "coordinate", "ancestry_trajectory.pdf", "FAS coordinate")
    plot_grouped("ancestry_profile.csv", "domain", "ancestor", "coordinate", "ancestry_profile.pdf", "FAS coordinate")
    plot_probe_types()
    write_narrative(main_df, scaling, validity, openset, sampling, geometry, encoder)
    print("Generated tables, available plots, and results/generated_narrative.tex")


if __name__ == "__main__":
    main()
