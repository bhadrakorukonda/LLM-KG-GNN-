from pathlib import Path
import json
p=Path('src/data/toy'); p.mkdir(parents=True, exist_ok=True)

nodes=[
    ('A','Alice is a researcher who collaborates with Bob on GraphRAG.'),
    ('B','Bob co-authored a paper with Carol about Knowledge Graphs.'),
    ('C','Carol maintains a dataset about citation networks.'),
    ('D','Dave focuses on edge-aware attention and EGAT.'),
    ('E','Eve curates Q&A benchmarks for graph-grounded QA.')
]
with open(p/'node_texts.jsonl','w',encoding='utf-8') as f:
    for nid,txt in nodes:
        f.write(json.dumps({'id':nid,'text':txt})+'\n')

edges=[('A','collaborates_with','B'),('B','coauthored_with','C'),
       ('C','advises','D'),('D','influences','E'),('A','mentions','D')]
with open(p/'kg_edges.tsv','w',encoding='utf-8') as f:
    for s,r,t in edges:
        f.write(f"{s}\t{r}\t{t}\n")

qas=[
    {"question":"Who does Alice collaborate with on GraphRAG?","answer":"Bob"},
    {"question":"Who co-authored a paper with Carol?","answer":"Bob"},
    {"question":"Who focuses on EGAT?","answer":"Dave"}
]
with open(p/'qa_dev.jsonl','w',encoding='utf-8') as f:
    for q in qas:
        f.write(json.dumps(q)+'\n')

print('Toy data written to src/data/toy/')
