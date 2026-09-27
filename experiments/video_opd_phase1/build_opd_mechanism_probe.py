"""Freeze the 128 previously sampled Base video trajectories for paired scoring."""

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from analyze_reasoning_qualification import partial_regions, token_spans, classify
from reasoning_format import parse_answer, parse_regions

BASE = Path(__file__).resolve().parent
STUDENT = "/mnt/data/zhzhu/models/Qwen3-VL-4B-Instruct"


def main():
    source = Path((BASE / "latest_reasoning_qual_run.txt").read_text().strip()) / "diagnostics"
    files = sorted(source.glob("step_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    assert len(files) == 8, files
    rows = {r["sample_id"]: r for r in map(json.loads, (BASE / "reasoning_qualification_128.jsonl").read_text().splitlines())}
    tok = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    out = {}
    for path in files:
        for sample in json.loads(path.read_text())["samples"]:
            sid = sample["sample_id"]
            assert sid not in out and sid in rows
            ids = [t["token_id"] for t in sample["tokens"]]
            no_eos = list(ids)
            while no_eos and no_eos[-1] in tok.all_special_ids:
                no_eos.pop()
            response, spans = token_spans(tok, no_eos)
            letters = [chr(ord("A") + i) for i in range(len(rows[sid]["options"]))]
            regions = parse_regions(response, letters) or partial_regions(response)
            labels = [classify(span, regions) for span in spans]
            labels += ["eos" if tid in tok.all_special_ids else "unclassified" for tid in ids[len(no_eos):]]
            assert len(ids) == len(labels)
            out[sid] = {
                "sample_id": sid, "original_video_id": rows[sid]["original_video_id"],
                "trajectory_token_ids": ids, "trajectory_response": response,
                "token_regions": labels, "response_tokens": len(ids),
                "reasoning_tokens": labels.count("reasoning"), "answer_tokens": labels.count("answer"),
                "parsed_answer": parse_answer(response, letters),
                "trajectory_source": str(path),
            }
    assert len(out) == 128 and set(out) == set(rows)
    output = BASE / "opd_mechanism_probe_128.jsonl"
    output.write_text("".join(json.dumps(out[sid], ensure_ascii=False) + "\n" for sid in rows))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    (BASE / "opd_mechanism_probe_128.sha256").write_text(f"{digest}  {output.name}\n")
    print(f"probe={output} N={len(out)} sha256={digest}")


if __name__ == "__main__":
    main()
