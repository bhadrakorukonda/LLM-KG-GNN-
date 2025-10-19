import argparse, json, pandas as pd, networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import re

def enumerate_paths(G, starts, max_hops=2, top_n=30):
    cands = []
    for s in starts:
        if s not in G: continue
        for t in G.nodes():
            if s==t: continue
            for path in nx.all_simple_paths(G, s, t, cutoff=max_hops):
                rels = [G.edges[path[i], path[i+1]]['rel'] for i in range(len(path)-1)]
                text = ' '.join([f'{path[i]} --{rels[i]}--> {path[i+1]}' for i in range(len(rels))]) or str(path[0])
                cands.append({'path': [str(x) for x in path], 'rels': [str(x) for x in rels], 'text': text})
    # de-dup
    seen, uniq = set(), []
    for c in cands:
        if c['text'] in seen: continue
        uniq.append(c); seen.add(c['text'])
    return uniq[:top_n]

def smart_start_nodes(G, question):
    # exact word tokens (Title Case to match your node casing)
    tokens = {t.strip('?,.!').title() for t in question.split()}
    exact = list(tokens & set(G.nodes()))
    if exact:
        return exact
    # phrase match: pick any node whose name appears in the question (case-insensitive)
    qlower = question.lower()
    phrase_hits = [n for n in G.nodes() if n.lower() in qlower]
    if phrase_hits:
        return phrase_hits
    # fallback: first node
    return list(G.nodes())[:1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kg', default='data/kg_edges.tsv')
    ap.add_argument('--node_texts', default='data/node_texts.jsonl')
    ap.add_argument('--qa', default='data/qa_dev.jsonl')
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--dump', default='baseline_paths.jsonl')  # JSONL output for re-ranker
    args = ap.parse_args()

    # clean read (no FutureWarning; drop blanks)
    df = pd.read_csv(args.kg, sep='\t', header=None, names=['subj','rel','obj'], dtype=str)
    for c in df.columns: df[c] = df[c].fillna('').str.strip()
    df = df[(df["subj"]!="") & (df["rel"]!="") & (df["obj"]!="")]

    G = nx.DiGraph()
    for s,r,o in df.itertuples(index=False):
        G.add_edge(s, o, rel=r)

    node_text = {}
    try:
        with open(args.node_texts, encoding='utf-8-sig') as f:
            for line in f: 
                j=json.loads(line); node_text[str(j["id"])] = j.get('text','')
    except FileNotFoundError:
        pass

    model = SentenceTransformer('all-MiniLM-L6-v2')

    def score_paths(q, paths):
        q_emb = model.encode([q])[0]
        texts = []
        for p in paths:
            items = []
            for i in range(len(p['path'])):
                n = p['path'][i]
                items.append(n)
                if n in node_text: items.append(node_text[n])
                if i < len(p['rels']):
                    items.append(p['rels'][i])
            texts.append(' '.join([t for t in items if t]))
        P = model.encode(texts)
        sims = P @ q_emb / (np.linalg.norm(P,axis=1)*np.linalg.norm(q_emb)+1e-9)
        return sims

    # truncate dump file
    open(args.dump, 'w').close()

    with open(args.qa, encoding='utf-8-sig') as f:
        for line in f:
            j = json.loads(line); q=j['q']
            starts = smart_start_nodes(G, q)
            paths = enumerate_paths(G, starts)
            if not paths:
                print("\nQ:", q); print("  (no paths found)"); continue
            sims = score_paths(q, paths)
            order = np.argsort(-sims)
            top = [{**paths[i], 'score': float(sims[i])} for i in order[:args.topk]]

            print("\nQ:", q)
            for i,t in enumerate(top,1):
                print("  {:>2}. {}  (score={:.3f})".format(i, t["text"], t["score"]))

            with open(args.dump, 'a', encoding='utf-8') as fout:
                fout.write(json.dumps({"q": q, "paths": top}, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()
