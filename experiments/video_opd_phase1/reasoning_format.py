"""Prompt and strict answer parser for reasoning Video OPD."""

import re


def reasoning_prompt(row):
    letters = ", ".join(chr(ord("A") + i) for i in range(len(row["options"])))
    return (
        "<video>\n"
        + row["question"] + "\n" + "\n".join(row["options"])
        + "\nReason briefly from the video evidence relevant to this question. "
        "Mention the pertinent people, objects, actions, timing, spatial relations, or changes when useful. "
        "Avoid generic guesses and filler. Then give one final option letter. "
        "Use this exact format:\n"
        "<thinking>Your concise video-based reasoning.</thinking>\n"
        "<answer>LETTER</answer>\n"
        f"Replace LETTER with exactly one of: {letters}."
    )


def parse_answer(response, allowed_letters):
    letters = "".join(re.escape(letter) for letter in allowed_letters)
    match = re.search(rf"<answer>\s*([{letters}])\s*</answer>", response)
    if match:
        return match.group(1)
    # A narrowly bounded fallback for occasional missing closing thinking tags.
    fallback = re.search(rf"(?:^|\n)Answer:\s*([{letters}])\s*$", response)
    return fallback.group(1) if fallback else None


def parse_regions(response, allowed_letters):
    """Return char spans only for a complete, unambiguous XML-like response."""
    letters = "".join(re.escape(letter) for letter in allowed_letters)
    match = re.fullmatch(
        rf"\s*(<thinking>)(.*?)(</thinking>)\s*(<answer>)\s*([{letters}])\s*(</answer>)\s*",
        response, flags=re.DOTALL,
    )
    if not match:
        return None
    return {
        "thinking_open": match.span(1),
        "reasoning": match.span(2),
        "thinking_close": match.span(3),
        "answer_open": match.span(4),
        "answer": match.span(5),
        "answer_close": match.span(6),
    }
