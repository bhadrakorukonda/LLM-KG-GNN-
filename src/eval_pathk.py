import argparse, json, re
from collections import defaultdict

def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def load_answers(qa_file):
    ans = {}
    with open(qa_file, encoding="utf-8-sig") as f:
        for line in f:
            j = json.loads(line)
            ans[j["q"]] = [norm(a) for a in j.get("answers", [])]
    return ans

def precision_at_k(paths_file, answers, k=5):
    total, hit = 0, 0
    per_q = {}
    with open(paths_file, encoding="utf-8-sig") as f:
        for line in f:
            item = json.loads(line)
            q = item["q"]
            golds = answers.get(q, [])
            if not golds: 
                continue
            total += 1
            found = False
            for i, p in enumerate(item["paths"][:k]):
                txt = norm(p["text"])
                if any(g in txt for g in golds):
                    found = True
                    break
            hit += int(found)
            per_q[q] = int(found)
    return (hit/total if total else 0.0), total, per_q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/qa_dev.jsonl")
    ap.add_argument("--baseline", default="baseline_paths.jsonl")
    ap.add_argument("--edgeaware", default="edgeaware_paths.jsonl")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    answers = load_answers(args.qa)
    p_baseline, n_b, _ = precision_at_k(args.baseline, answers, args.k)
    p_edge,     n_e, _ = precision_at_k(args.edgeaware, answers, args.k)

    print(f"Questions evaluated: {max(n_b, n_e)}")
    print(f"Precision@{args.k} — baseline: {p_baseline:.3f}  edge-aware: {p_edge:.3f}  (Δ={p_edge - p_baseline:+.3f})")

if __name__ == "__main__":
    main()
