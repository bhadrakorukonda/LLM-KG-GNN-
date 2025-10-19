import argparse, json

# simple handcrafted priors; tweak per your data
REL_WEIGHT = {
    "used_for": 1.1,
    "enhances": 1.2,
    "consumes": 1.0,
    "contains": 1.05,
    "connected_by": 0.95,
}

def edge_score(rels):
    s = 1.0
    for r in rels:
        s *= REL_WEIGHT.get(r, 1.0)
    # prefer shorter paths slightly
    if len(rels) > 1:
        s /= (1.0 + 0.15*(len(rels)-1))
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_paths", default="baseline_paths.jsonl")
    ap.add_argument("--out_paths", default="edgeaware_paths.jsonl")
    ap.add_argument("--lambda_", type=float, default=0.7)  # blend baseline score with edge prior
    args = ap.parse_args()

    with open(args.out_paths, "w", encoding="utf-8") as fout:
        for line in open(args.in_paths, encoding="utf-8-sig"):
            item = json.loads(line)
            rescored=[]
            for p in item["paths"]:
                ea = edge_score(p.get("rels",[]))
                new = args.lambda_*p["score"] + (1-args.lambda_)*ea
                q = dict(p); q["edgeaware_score"]=float(new)
                rescored.append(q)
            rescored.sort(key=lambda x: -x["edgeaware_score"])
            fout.write(json.dumps({"q": item["q"], "paths": rescored}, ensure_ascii=False) + "\n")
    print("Wrote:", args.out_paths)

if __name__ == "__main__":
    main()
