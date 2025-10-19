import argparse, pandas as pd, networkx as nx, io

def load_edges(path):
    # Read with BOM tolerance and flexible separator (tab OR 2+ spaces)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        txt = f.read()
    # Try pandas with tab; fallback to whitespace split
    try:
        df = pd.read_csv(io.StringIO(txt), sep="\t", header=None,
                         names=["subj","rel","obj"], dtype=str)
        if df.isna().all(axis=None):
            raise ValueError("empty after tab read")
    except Exception:
        df = pd.read_csv(io.StringIO(txt), sep=r"\s{2,}", engine="python",
                         header=None, names=["subj","rel","obj"], dtype=str)
    # strip
    df = df.astype(str)
    for c in df.columns:
        df[c] = df[c].str.strip()
    # drop any blank rows
    df = df[(df["subj"]!="") & (df["rel"]!="") & (df["obj"]!="")]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="data/kg_edges.tsv")
    args = ap.parse_args()

    df = load_edges(args.kg)
    G = nx.DiGraph()
    for s,r,o in df.itertuples(index=False):
        G.add_edge(s, o, rel=r)
    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    shown = 0
    for s in G.nodes():
        for t in G.nodes():
            if s==t: continue
            try:
                for path in nx.all_simple_paths(G, s, t, cutoff=2):
                    labs = [G.edges[path[i], path[i+1]]['rel'] for i in range(len(path)-1)]
                    print('PATH:', ' -> '.join(map(str, path)), '| rels:', ' -> '.join(map(str, labs)))
                    shown += 1
                    if shown>=5: raise StopIteration
            except (nx.NetworkXNoPath, StopIteration):
                break
        if shown>=5: break

if __name__ == "__main__":
    main()
