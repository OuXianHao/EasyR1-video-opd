"""Render the reproducible Phase 1.7 diagnostic report from measured JSON."""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
R = json.loads((BASE / 'phase1_7_report.json').read_text())
S = R['smokes']
D = R['decision']
M = R['model_metrics']


def pct(x):
    return f'{100*x:.1f}%'


def pair(a, b, digits=1):
    return f'{a:.{digits}f}/{b:.{digits}f}'


lines = []


def add(*items):
    lines.extend(items)


add(
    '# Video OPD Phase 1.7: corrected preprocessing, held-out 500, pilot decision',
    '',
    '## Decision',
    '',
    '```text',
    f"PREPROCESSING_FIXED = {D['PREPROCESSING_FIXED']}",
    f"RECOMMENDED_VISUAL_BUDGET = {D['RECOMMENDED_VISUAL_BUDGET']}",
    f"TEACHER_32B = {D['TEACHER_32B']}",
    f"DATASET = {D['DATASET']}",
    f"READY_FOR_1K_OPD = {D['READY_FOR_1K_OPD']}",
    '```',
    '',
    'The engineering path is ready for a **bounded research pilot**. The 32B Teacher has a small and statistically uncertain overall accuracy lead, a worse mean GT NLL caused by some severe low-GT-probability errors, and stronger evidence on selected types, especially Action Count. This is partial qualification, not a claim of broad Teacher dominance. No 1k/3k training was started in this turn.',
    '',
    '## 1. Preprocessing fix and common input',
    '',
    'The former frame-list path in `qwen-vl-utils` applied a low 16,384-pixel per-frame cap and its list branch effectively used a 56-pixel resize factor before the Transformers video processor rounded again. Typical 1280×720 frames became 64×160, grid `[16,4,10]`, only 160 merged visual tokens. The new Video OPD mode (`EASYR1_VIDEO_OPD_PREPROCESS=1`) decodes the existing ordered 32 JPGs, applies **one aspect-preserving PIL resize**, centers the result on a 32-pixel-aligned neutral canvas, and supplies the same tensor to rollout and scorers. `qwen-vl-utils` is bypassed for these frame lists. The Transformers/Qwen video processor is called with `do_resize=False` for normalization and patchification; vLLM receives the same setting through `mm_processor_kwargs`. The legacy path remains available when the new mode is not enabled.',
    '',
    'The target canvas never exceeds the source frame width or height; low-resolution sources are not enlarged. All 500 qualification videos have 32 source frames of uniform within-video size. Debug mode prints source resolution, resized content, canvas resolution, grid and visual-token count. The Student and Teacher processor configurations match; scorer visual tensors are asserted equal. The very small vLLM–FSDP sampled-token logprob differences in both real smokes provide an additional cross-path check.',
    '',
    '| Candidate | Per-frame canvas area cap | Typical source → content → canvas | Grid | Tokens |',
    '|---|---:|---|---|---:|',
    '| Medium | 86,016 pixels | 1280×720 → 384×216 → 384×224 (center padding) | `[16,14,24]` | 1,344 |',
    '| High | 147,456 pixels | 1280×720 → 512×288 → 512×288 | `[16,18,32]` | 2,304 |',
    '',
    'The independent candidate files are `medium.env` and `high.env`; `run_phase1_7_smoke.sh` records the effective runtime command. Both preserve the same 32 frames, question, options, one-letter answer format and OPD loss.',
    '',
    '## 2. Real one-step OPD costs',
    '',
    'Both conditions used the same two qualification samples, 4B Student, sharded 32B Teacher, two PPUs, `n=1`, one global step, and one actual backward/optimizer step. Times are measured inside the step, excluding model initialization. Generation time includes rollout-engine wake and release overhead. External memory comes from 2-second `nvidia-smi` samples, so its peak is approximate.',
    '',
    '| Measure | Medium | High | High / Medium |',
    '|---|---:|---:|---:|',
    f"| Visual tokens per video | {S['medium']['visual_tokens']:,} | {S['high']['visual_tokens']:,} | {S['high']['visual_tokens']/S['medium']['visual_tokens']:.2f}× |",
    f"| Student rollout / generation | {S['medium']['student_rollout_time_s']:.2f} s | {S['high']['student_rollout_time_s']:.2f} s | {S['high']['student_rollout_time_s']/S['medium']['student_rollout_time_s']:.2f}× |",
    f"| Old scorer | {S['medium']['old_scorer_time_s']:.2f} s | {S['high']['old_scorer_time_s']:.2f} s | {S['high']['old_scorer_time_s']/S['medium']['old_scorer_time_s']:.2f}× |",
    f"| Teacher scorer | {S['medium']['teacher_scorer_time_s']:.2f} s | {S['high']['teacher_scorer_time_s']:.2f} s | {S['high_over_medium_teacher_time_ratio']:.2f}× |",
    f"| Backward | {S['medium']['backward_time_s']:.2f} s | {S['high']['backward_time_s']:.2f} s | {S['high']['backward_time_s']/S['medium']['backward_time_s']:.2f}× |",
    f"| Total step | {S['medium']['step_time_s']:.2f} s | {S['high']['step_time_s']:.2f} s | {S['high_over_medium_step_time_ratio']:.2f}× |",
    f"| PPU 0 peak | {S['medium']['peak_external_memory_mib_per_ppu']['0']:,} MiB | {S['high']['peak_external_memory_mib_per_ppu']['0']:,} MiB | — |",
    f"| PPU 1 peak | {S['medium']['peak_external_memory_mib_per_ppu']['1']:,} MiB | {S['high']['peak_external_memory_mib_per_ppu']['1']:,} MiB | — |",
    f"| Maximum peak-memory ratio | — | — | {S['high_over_medium_peak_memory_ratio']:.3f}× |",
    f"| Grad norm | {S['medium']['grad_norm']:.6f} | {S['high']['grad_norm']:.6f} | — |",
    f"| OPD loss | {S['medium']['opd_loss']:+.7f} | {S['high']['opd_loss']:+.7f} | — |",
    '',
    'Both optimizer-step flags equal 1. Student/Teacher scorer grids and pixels were equal in both runs. Medium and High respectively had vLLM–FSDP absolute sampled-token logprob mean 1.52e-5 and 7.35e-6 (four response tokens each). High adds only 3.3% to this single measured step and no observed external-memory increase; its Teacher scoring is 16.9% slower. The budget recommendation is **High**. A one-step ratio is a cost probe, not a long-run throughput estimate.',
    '',
    '## 3. Fixed, video-disjoint qualification and train pool',
    '',
    'The local 32-frame RLSD-V package has 14,511 QA across 760 original-video IDs. The old train/val split overlapped on original videos, so this turn froze a new split by original-video ID. `qualification_500.jsonl` contains exactly one QA from each of 500 originals, with all 27 question types. Type quotas are the largest-remainder approximation to the full 14,511-QA type distribution; within-type choices use a fixed SHA-256 salt. No 4B/32B answer, confidence or tag was used in selection. All sample metadata and the existing ordered JPG paths are retained. The future train pool contains the other 260 originals and 4,145 QA: **zero original-video overlap** with qualification. No sample was deleted from the source package; all QA from reserved evaluation originals are excluded from the future train pool to enforce this boundary.',
    '',
    f"The new qualification set has {R['prior_diagnostic_overlap']['overlap_count']} original videos in common with the earlier diagnostic_64; {R['prior_diagnostic_overlap']['fresh_count']} are new relative to that budget-selection subset. The selection did not use prior outcomes, and fresh-only results are reported below. All 710 distinct train-pool clips were checked at the JPG-header level: zero missing frames, non-32-frame clips, or mixed-resolution clips. The train pool's longest raw text prompt was 538 tokens; its ten longest prompts were actually checked with High video, with processed lengths 2,828–2,973, below the candidate `max_prompt_length=3072`.",
    '',
    '## 4. High-budget Teacher qualification on 500',
    '',
    'Each model generated a direct option letter deterministically (`do_sample=False`, `max_new_tokens=8`) with the same question/options. The exact GT letter next-token log probability was scored against the **full vocabulary**; `GT NLL = −logP(GT)`, and margin is GT log probability minus the best wrong option-letter log probability. Text-only removes only the video content block. All 2,000 answers across model × condition were format valid. Both processors produced identical High grid/resolution for every paired video input.',
    '',
    '| Model | Video correct | Text-only correct | Video GT logP | Text GT logP | Video GT NLL | Video GT prob | Video margin | Mean visual gain |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
)
for model in ('4b', '32b'):
    v = M[model]['video']
    t = M[model]['text_only']
    add(f"| {model.upper()} | {v['correct']}/500 ({pct(v['accuracy'])}) | {t['correct']}/500 ({pct(t['accuracy'])}) | {v['mean_gt_logprob']:.3f} | {t['mean_gt_logprob']:.3f} | {v['mean_gt_nll']:.3f} | {v['mean_gt_probability']:.3f} | {v['mean_margin']:.3f} | {R['visual_gain'][model]['mean']:+.3f} |")

paired = R['video_paired_4b_32b']
fresh = R['prior_diagnostic_overlap']['fresh_only_metrics']
add(
    '',
    f"Paired video answers: {paired['both_correct']} both correct, {paired['first_only']} 4B only, {paired['second_only']} 32B only, {paired['both_wrong']} both wrong. The 32B accuracy lead is **+1.8 percentage points**; paired bootstrap 95% interval is **{pct(R['video_accuracy_gap_bootstrap_95_ci'][0])} to {pct(R['video_accuracy_gap_bootstrap_95_ci'][1])}**. On the 452 samples outside the previous diagnostic_64, 4B/32B are {pct(fresh['4b_video_acc'])}/{pct(fresh['32b_video_acc'])}, a +1.55-point gap whose interval also crosses zero. Thus the overall advantage is small and uncertain.",
    '',
    'The 32B has a higher mean GT probability and a larger margin, but **worse mean GT NLL** (0.863 versus 0.676) because a minority of wrong answers assign the GT much lower probability. Its GT log probability exceeds 4B on 357/500 rows, while the negative tail reaches about −8.93 log-prob units in the 32B−4B comparison. This is a real Teacher-quality limitation for OPD; neither mean margin nor accuracy alone should hide it.',
    '',
    'Video materially helps this held-out set: 4B has 77 more correct video answers than text-only (+15.4 percentage points; 399 versus 322), and 32B has 55 more (+11.0 points; 408 versus 353). Paired video-only versus text-only correct counts are 108 versus 31 for 4B, and 94 versus 39 for 32B. Mean GT log-prob visual gains are +1.089 (4B) and +0.555 (32B). Medians are near zero because many already-easy rows change little; the positive means come from visually useful rows. High produced 2,304 visual tokens on 475/500 samples (median 2,304, minimum 960 due to source resolution), and the maximum input length was 2,807.',
    '',
    '## 5. Question types and analysis tags',
    '',
    'The table shows **all 27 existing types**, including train-pool QA counts. Accuracies and all other requested metrics use the fixed 500; `V` and `T` mean video and text-only. Gain is `logP(GT|video,q) − logP(GT|q)`. Paired entries are 4B/32B. A type with only 2–8 evaluation rows is descriptive, not a stable ranking.',
    '',
    '| Question type | Eval N | Pool QA | V acc 4B/32B | T acc 4B/32B | Gap | Gain 4B/32B | NLL 4B/32B | Margin 4B/32B |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
)
pool_counts = R['train_pool_distribution']['type_counts']
for type_name, x in sorted(R['type_metrics'].items(), key=lambda item: (-item[1]['n'], item[0])):
    add(
        f"| {type_name} | {x['n']} | {pool_counts[type_name]} | "
        f"{pct(x['4b_video_acc'])}/{pct(x['32b_video_acc'])} | "
        f"{pct(x['4b_text_acc'])}/{pct(x['32b_text_acc'])} | "
        f"{100*x['32b_minus_4b_video_acc']:+.1f} pp | "
        f"{pair(x['mean_visual_gain_4b'],x['mean_visual_gain_32b'],2)} | "
        f"{pair(x['4b_gt_nll'],x['32b_gt_nll'],2)} | "
        f"{pair(x['4b_margin'],x['32b_margin'],2)} |"
    )

add(
    '',
    'At family level, Objects has 135 evaluation rows: 4B 95/135 and 32B 97/135 video-correct, versus 65/135 and 68/135 text-only. The strong Objects Teacher accuracy advantage seen on the old 14-row diagnostic **did not replicate**. On the 122 Objects rows outside diagnostic_64, both models are exactly 90/122 video-correct. Objects is video-dependent, but not broadly Teacher-resolvable in this larger sample.',
    '',
    'Action Count is the clearest Teacher-resolvable major type: 32B 12/24 versus 4B 6/24 with video, paired 6 both correct / 0 4B only / 6 32B only / 12 both wrong. Both text-only scores are 8/24; 32B gains from video while 4B does not. The 32B also has lower GT NLL (1.84 versus 2.22) and positive mean visual gain (+0.77). On the 19 Action Count rows not seen in diagnostic_64, 32B is 9/19 versus 4B 4/19. Objects Spatial Relation/Location and Event Sequence show smaller potential signals; Action Location is hard for both and 32B is weaker there. Brief/Detailed Description are near-saturated (73/74 correct for each model with video); the 32B already answers 72/74 correctly without video.',
    '',
    'Each qualification sample has nonexclusive **analysis-only** tags in `phase1_7_qualification_tagged.jsonl`: easy/text-prior-heavy 243, video-dependent 158, Teacher-resolvable 33, both-hard 68, plus 29 unclassified. Easy means both models are correct in both modalities and each |visual gain|≤0.5; video-dependent means a video-only correct answer with positive gain >0.25, or both gains >0.5; Teacher-resolvable means 4B video wrong, 32B video correct and higher GT probability; both-hard means both video wrong and both GT probabilities <0.5. Tags can overlap and **do not enter OPD loss or evaluation selection**.',
    '',
    '## 6. Future 1k research pilot composition',
    '',
    'The proposal below uses **only** the 4,145-QA train pool, draws 1,000 unique QA without replacement, and caps at eight QA per original video. A deterministic feasibility dry run filled all buckets with 1,000 distinct QA across 241 originals and no video exceeding the cap; it did not save a 1k manifest or start training. Draw the 400-row base proportional to the train-pool type distribution first, then fill the named boosts and coverage bucket without replacement. Preserve the frozen, outcome-independent `qualification_500` as the evaluation distribution.',
    '',
    '| Bucket | Proposed QA | Rationale |',
    '|---|---:|---|',
    '| Train-pool type-proportional base | 400 | Preserve broad dataset coverage |',
    '| Action Count | 130 | Largest reproducible 32B advantage; train pool has 196 QA across 115 originals |',
    '| Objects Spatial Relation | 60 | Strong visual dependence, small 32B lead |',
    '| Objects Spatial Location | 50 | Visual dependence and some Teacher-resolvable rows |',
    '| Objects Existence | 70 | Strong video signal despite little Teacher accuracy gap |',
    '| Event Sequence | 50 | Modest Teacher signal; retains temporal questions |',
    '| Remaining-type coverage | 240 | Include rare and hard types without concentrating easy descriptions |',
    '| **Total** | **1,000** | |',
    '',
    'This proposed oversampling is for **training only**. Do not filter or reweight the held-out 500 by Teacher correct/Student wrong, by analysis tag, or by any pilot result. In a later turn, materialize and audit the 1k manifest before starting training.',
    '',
    '## Files and limits',
    '',
    '- Runtime paths: `medium.env`, `high.env`, `run_phase1_7_smoke.sh`; measured one-step runs are linked by `latest_phase1_7_medium_run.txt` and `latest_phase1_7_high_run.txt`.',
    '- Fixed evaluation: `qualification_500.jsonl`; disjoint pool: `train_pool_video_disjoint.jsonl`; split audit: `qualification_500_split_audit.json`.',
    '- Raw deterministic answer/log-prob records: `phase1_7_{4b,32b}_high_text_only.jsonl` and `phase1_7_{4b,32b}_high_video_p*.jsonl`. Per-sample tags are in `phase1_7_qualification_tagged.jsonl`; all aggregates and exact-type metrics are in `phase1_7_report.json`. The proposed 1k quota dry run is recorded in `pilot_quota_feasibility.json`.',
    '- The local package is named RLSD-V frames32; prior file inspection could not establish a separate BSPO provenance. The 32-frame assets are complete. No new data was downloaded or constructed.',
    "- The 32B's overall +1.8-point accuracy gap is uncertain, its mean GT NLL is worse, and the Objects family gap did not replicate. `PARTIALLY_QUALIFIED` and `READY_FOR_1K_OPD=YES` mean that a bounded, stratified **research pilot is technically justified**, not that OPD improvement is predicted or that a larger run is authorized.",
    '',
)

(BASE / 'phase1_7_report.md').write_text('\n'.join(lines))
print(BASE / 'phase1_7_report.md')
