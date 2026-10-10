"""
Optional LLM-as-judge pass over saved eval runs.

Grades each answer 0 / 1 / 2 against the reference WITHOUT telling the judge
which model produced it (blinded):
  2 = correct (conveys the reference's key point, no significant errors)
  1 = partially correct (relevant but incomplete or partly wrong)
  0 = wrong, off-topic, or a refusal

Needs ANTHROPIC_API_KEY and `pip install anthropic`. Set JUDGE_MODEL to
override the default model. Scores are written back into each run file as
`judge_score`, which src/report.py picks up.

Usage: python src/judge.py eval/runs/*.json
"""
import os
import re
import sys
import json

PROMPT = """You are grading an answer to a machine-learning question against a reference answer written from the source paper.

Question:
{question}

Reference answer:
{reference}

Candidate answer:
{answer}

Score the candidate:
2 = correct: conveys the reference's key point with no significant errors (extra correct detail is fine)
1 = partially correct: relevant but incomplete, vague, or containing a significant error alongside correct content
0 = wrong, off-topic, or a refusal

Reply with a single line of JSON only, like {{"score": 1, "reason": "<one short sentence>"}}"""


def grade(client, model, item):
    msg = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": PROMPT.format(
            question=item["question"], reference=item["reference"], answer=item["answer"])}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    data = json.loads(m.group(0)) if m else {}
    score = int(data.get("score", -1))
    if score not in (0, 1, 2):
        raise ValueError(f"unparseable judge reply: {text[:120]!r}")
    return score, data.get("reason", "")


def main(paths):
    import anthropic
    client = anthropic.Anthropic()
    model = os.environ.get("JUDGE_MODEL", "claude-sonnet-5-5")
    for path in paths:
        with open(path) as f:
            run = json.load(f)
        if all("judge_score" in r for r in run["results"]):
            print(f"[skip] {path} already judged")
            continue
        for item in run["results"]:
            if "judge_score" in item:
                continue
            item["judge_score"], item["judge_reason"] = grade(client, model, item)
        with open(path, "w") as f:
            json.dump(run, f, indent=2)
        scores = [r["judge_score"] for r in run["results"]]
        print(f"[judged] {os.path.basename(path)}: mean {sum(scores)/len(scores):.2f}/2, "
              f"fully correct {sum(s == 2 for s in scores)}/{len(scores)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python src/judge.py eval/runs/*.json")
    main(sys.argv[1:])
