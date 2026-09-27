"""Render the numerical decomposition JSON to a readable pilot diagnostic report."""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
STUDENTS = ("base", "step16", "step31", "step47", "step62")
MODELS = STUDENTS + ("teacher",)


def f(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def main():
    data = json.loads((BASE / "opd_signal_decomposition.json").read_text())
    b, m = data["behavior"], data["mechanism"]
    lines = [
        "# 1K Natural Reasoning Video OPD: pilot diagnostic evaluation",
        "",
        "This is a fixed **pilot diagnostic set**, not a final test benchmark. No new training, loss change, entropy mechanism, or gradient attribution was used.",
        "",
        "## Sets and method",
        "",
        f"- Behavior: {data['datasets']['evaluation_n']} QA, {data['datasets']['evaluation_unique_original_videos']} original videos; training original-video overlap: {data['datasets']['training_original_video_overlap']}.",
        f"- Mechanism: {data['datasets']['mechanism_n']} fixed Base 4B video trajectories, frozen SHA-256 `{data['datasets']['mechanism_sha256']}`. Each model and condition scores the same response token IDs and prefix. The 128 were previously selected for High-budget source resolution before any model outcomes; no correctness-based selection.",
        "- Behavior uses greedy reasoning generation with the training prompt, up to 1024 response tokens. Video uses all 32 source frames and the production High visual preprocessing. Text condition uses the same question, options, and reasoning instruction without the video token.",
        "- Mechanism statistics pool response tokens; longer trajectories have more weight. Reasoning, answer, and format/EOS are also shown separately. The gap identity is an algebraic decomposition, not a causal attribution.",
        "- Pearson and Spearman correlations are computed on paired token values; the top-k selection uses absolute video OPD gap. No `abs(shared)/abs(OPD)` ratios are interpreted as contribution fractions.",
        "- All six tokenizers have the same vocabulary size and identical raw video/text chat prompt IDs on a fixed probe example. vLLM scorer values were cross-checked against the earlier FSDP qualification scorer on the same 128 video trajectories; differences reflect inference implementation and precision, not trajectory mismatch.",
        "",
        "Scorer cross-check, absolute logprob difference against qualification FSDP (`N=13,445` tokens):",
        "",
        "| Model | Mean | Median | P95 | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ("base", "teacher"):
        z = data["vllm_vs_qualification_fsdp_logprob_crosscheck"][model]
        lines.append(f"| {model} | {f(z['mean_absolute_difference'])} | {f(z['median_absolute_difference'])} | {f(z['p95_absolute_difference'])} | {f(z['max_absolute_difference'])} |")
    lines += [
        "",
        "## Behavioral evaluation",
        "",
        "| Model | Video acc | Text acc | Video−text | Video format valid | Text format valid | Video response median / mean / P95 | Text response median / mean / P95 | Video reasoning median | Text reasoning median | Video 1024 cap | Text 1024 cap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        v, t = b[model]["video"], b[model]["text"]
        vr, tr = v["response_tokens"], t["response_tokens"]
        lines.append(f"| {model} | {f(v['accuracy']*100,1)}% ({v['correct']}/500) | {f(t['accuracy']*100,1)}% ({t['correct']}/500) | {f(b[model]['video_minus_text_accuracy']*100,1)} pp | {f(v['format_valid_rate']*100,1)}% | {f(t['format_valid_rate']*100,1)}% | {f(vr['p50'],0)} / {f(vr['mean'],1)} / {f(vr['p95'],0)} | {f(tr['p50'],0)} / {f(tr['mean'],1)} / {f(tr['p95'],0)} | {f(v['reasoning_tokens']['p50'],0)} | {f(t['reasoning_tokens']['p50'],0)} | {v['truncated_count']} | {t['truncated_count']} |")
    lines += ["", "Video visual-token counts (source resolution is preserved; lower-resolution source clips stay lower):", ""]
    for model in MODELS:
        lines.append(f"- {model}: `{b[model]['video']['visual_token_counts']}`")
    lines += ["", "### Paired changes versus Base", "",
              "The same 500 QA are used for each model. Gained/lost count questions changing correctness relative to Base; exact two-sided McNemar p values describe sampling uncertainty on this pilot set.", "",
              "| Model | Video gained / lost / net | Video p | Text gained / lost / net | Text p |",
              "|---|---:|---:|---:|---:|"]
    for model in STUDENTS[1:]:
        v = data["paired_changes_vs_base"][model]["video"]
        t = data["paired_changes_vs_base"][model]["text"]
        lines.append(f"| {model} | {v['gained_correct_vs_base']} / {v['lost_correct_vs_base']} / {v['net']:+d} | {f(v['mcnemar_exact_two_sided_p'])} | {t['gained_correct_vs_base']} / {t['lost_correct_vs_base']} / {t['net']:+d} | {f(t['mcnemar_exact_two_sided_p'])} |")
    lines += [
        "", "## OPD signal decomposition", "",
        "For each fixed token, `OPD = logP(T|video) − logP(S|video)`, `shared = logP(T|text) − logP(S|text)`, and `visual = [logP(T|video) − logP(T|text)] − [logP(S|video) − logP(S|text)]`.",
        "", "The identity `OPD = shared + visual` is checked over every scored token:", "",
        f"- Mean absolute error: `{data['identity_error']['mean_absolute']:.3e}`; P99: `{data['identity_error']['p99']:.3e}`; max: `{data['identity_error']['max']:.3e}`; comparisons: `{data['identity_error']['count']}`.",
        "", "### Reasoning tokens: checkpoint dynamics", "",
        "| Model | Tokens | mean OPD | mean shared | mean visual | mean\\|OPD\\| | mean\\|shared\\| | mean\\|visual\\| | Pearson OPD/shared | Spearman OPD/shared | Pearson OPD/visual | Spearman OPD/visual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in STUDENTS:
        r = m[model]["reasoning"]
        s = r["stats"]
        c = r["correlation"]
        lines.append(f"| {model} | {s['opd']['count']} | {f(s['opd']['mean'])} | {f(s['shared']['mean'])} | {f(s['visual']['mean'])} | {f(s['opd']['mean_abs'])} | {f(s['shared']['mean_abs'])} | {f(s['visual']['mean_abs'])} | {f(c['opd_shared']['pearson'])} | {f(c['opd_shared']['spearman'])} | {f(c['opd_visual']['pearson'])} | {f(c['opd_visual']['spearman'])} |")
    opposite = ", ".join(f"{model}: {f(m[model]['reasoning']['shared_visual_opposite_sign_fraction']*100,1)}%" for model in STUDENTS)
    lines += ["", f"Shared and visual gaps have opposite signs on reasoning tokens ({opposite}). This cancellation explains why mean absolute OPD can be smaller than either component; their absolute values are not additive contributions."]
    lines += ["", "### Answer tokens", "",
              "| Model | Tokens | mean\\|OPD\\| | mean\\|shared\\| | mean\\|visual\\| | Pearson OPD/shared | Pearson OPD/visual |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for model in STUDENTS:
        r = m[model]["answer"]
        s, c = r["stats"], r["correlation"]
        lines.append(f"| {model} | {s['opd']['count']} | {f(s['opd']['mean_abs'])} | {f(s['shared']['mean_abs'])} | {f(s['visual']['mean_abs'])} | {f(c['opd_shared']['pearson'])} | {f(c['opd_visual']['pearson'])} |")
    lines += ["", "### Full region distributions", "",
              "Each line gives count, mean, mean absolute value, standard deviation, P5/P25/P50/P75/P95, and positive fraction for the three signals.", ""]
    for region in ("reasoning", "answer", "format", "eos"):
        lines += [f"#### {region}", "",
                  "| Model | Signal | N | mean | mean abs | std | P5 | P25 | P50 | P75 | P95 | positive |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for model in STUDENTS:
            if region not in m[model]:
                continue
            for name in ("opd", "shared", "visual"):
                s = m[model][region]["stats"][name]
                lines.append(f"| {model} | {name} | {s['count']} | {f(s['mean'])} | {f(s['mean_abs'])} | {f(s['std'])} | {f(s['p5'])} | {f(s['p25'])} | {f(s['p50'])} | {f(s['p75'])} | {f(s['p95'])} | {f(s['positive_fraction']*100,1)}% |")
        lines.append("")
    for region in ("reasoning", "answer", "all"):
        lines += [f"### Strong-token analysis: {region}", "",
                  "| Model | Top \\|OPD\\| | N | mean\\|OPD\\| | mean\\|shared\\| | mean\\|visual\\| | fraction \\|shared\\| > \\|visual\\| |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for model in STUDENTS:
            if region not in m[model]:
                continue
            for k in (1, 5, 10):
                s = m[model][region]["topk_abs_opd"][str(k)]
                lines.append(f"| {model} | {k}% | {s['count']} | {f(s['mean_abs_opd'])} | {f(s['mean_abs_shared'])} | {f(s['mean_abs_visual'])} | {f(s['shared_abs_gt_visual_fraction']*100,1)}% |")
        lines.append("")
    lines += ["## Behavior and mechanism together", "",
              "| Model | Video acc | Text acc | mean\\|OPD\\| reasoning | mean\\|shared\\| reasoning | mean\\|visual\\| reasoning | corr(OPD,shared) | corr(OPD,visual) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in STUDENTS:
        r = m[model]["reasoning"]
        lines.append(f"| {model} | {f(b[model]['video']['accuracy']*100,1)}% | {f(b[model]['text']['accuracy']*100,1)}% | {f(r['stats']['opd']['mean_abs'])} | {f(r['stats']['shared']['mean_abs'])} | {f(r['stats']['visual']['mean_abs'])} | {f(r['correlation']['opd_shared']['pearson'])} | {f(r['correlation']['opd_visual']['pearson'])} |")
    d = data["conclusions"]
    lines += ["", "## Conclusion", "", d["interpretation"], "",
              "| Decision | Result |", "|---|---|",
              *[f"| `{key}` | **{d[key]}** |" for key in (
                  "TASK_PERFORMANCE_IMPROVED", "OPD_SIGNAL_PRIMARILY_SHARED",
                  "VISUAL_SPECIFIC_GAP_SUBSTANTIAL", "GRADIENT_ATTRIBUTION_WORTH_RUNNING")],
              "", "The 500-QA results are pilot associations. The fixed 128 trajectories control response-prefix variation, but they do not establish a causal link between the decomposed logprob gaps and the training updates. Gradient attribution remains future work; it was not performed in this round.", ""]
    (BASE / "opd_signal_decomposition.md").write_text("\n".join(lines))
    print("Wrote", BASE / "opd_signal_decomposition.md")


if __name__ == "__main__":
    main()
