"""Read-only token-region diagnostics for reasoning OPD rollouts."""

import re


def _spans(tokenizer, ids):
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    pieces = tokenizer.batch_decode([[i] for i in ids], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
    if "".join(pieces) != text:
        prefixes = [""] + [tokenizer.decode(ids[:i], skip_special_tokens=False,
                    clean_up_tokenization_spaces=False) for i in range(1, len(ids) + 1)]
        return text, [(len(prefixes[i]), len(prefixes[i + 1])) for i in range(len(ids))]
    offsets = [0]
    for piece in pieces:
        offsets.append(offsets[-1] + len(piece))
    return text, list(zip(offsets[:-1], offsets[1:]))


def _regions(text):
    start = text.find("<thinking>")
    if start < 0:
        return {}
    content_start = start + len("<thinking>")
    ends = [pos for marker in ("</thinking>", "<answer>", "\nAnswer:")
            if (pos := text.find(marker, content_start)) >= 0]
    regions = {"reasoning": (content_start, min(ends) if ends else len(text))}
    answer = re.search(r"<answer>\s*([A-Z])", text)
    if answer:
        regions["answer"] = answer.span(1)
    return regions


def analyze_response(tokenizer, token_ids, deltas=None):
    """Classify sampled tokens; a token joined with '>A' counts as answer."""
    ids = list(token_ids)
    text_ids = list(ids)
    while text_ids and text_ids[-1] in tokenizer.all_special_ids:
        text_ids.pop()
    text, spans = _spans(tokenizer, text_ids)
    regions = _regions(text)
    labels = []
    for a, b in spans:
        label = "format_or_other"
        for name in ("answer", "reasoning"):
            if name in regions:
                x, y = regions[name]
                if max(0, min(b, y) - max(a, x)) > 0:
                    label = name
                    break
        labels.append(label)
    labels.extend("eos" if tid in tokenizer.all_special_ids else "format_or_other"
                  for tid in ids[len(text_ids):])
    if len(labels) != len(ids):
        raise ValueError("Reasoning diagnostic token alignment failed")
    answer = re.search(r"<answer>\s*([A-Z])\s*</answer>", text)
    if not answer:
        answer = re.search(r"(?:^|\n)Answer:\s*([A-Z])\s*$", text)
    result = {"response_text": text, "final_answer": answer.group(1) if answer else None,
              "response_tokens": len(ids), "reasoning_tokens": labels.count("reasoning"),
              "answer_tokens": labels.count("answer"), "labels": labels,
              "has_thinking_open": "<thinking>" in text,
              "has_thinking_close": "</thinking>" in text,
              "has_answer_tag": "<answer>" in text and "</answer>" in text}
    if deltas is not None:
        if len(deltas) != len(ids):
            raise ValueError("Reasoning diagnostic Delta length mismatch")
        result["delta_by_region"] = {}
        for name in ("reasoning", "answer", "format_or_other", "eos"):
            values = [float(value) for value, label in zip(deltas, labels) if label == name]
            result["delta_by_region"][name] = {
                "count": len(values),
                "mean": sum(values) / len(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "positive_fraction": sum(value > 0 for value in values) / len(values) if values else None,
            }
    return result
