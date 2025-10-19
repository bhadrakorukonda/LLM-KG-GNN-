from backend.services.graph import graph_store
from backend.services.paths import find_paths, neighborhood

def test_find_paths_small(tmp_path):
    edges = tmp_path / "e.tsv"
    nodes = tmp_path / "n.jsonl"
    edges.write_text("Carol\tcoauthored_with\tBob\nBob\tcoauthored_with\tCarol\n", encoding="utf-8")
    nodes.write_text('{"id":"Carol","text":"Carol is a PhD student; co-authored with Bob."}\n{"id":"Bob","text":"Bob collaborates with Alice and Carol; co-authored with Carol."}\n', encoding="utf-8")

    stats = graph_store.reload(str(edges), str(nodes))
    assert stats["edges"] == 2

    paths = find_paths(["Carol"], max_hops=1, topk_paths=5)
    assert paths, "Expected at least one 1-hop path from Carol"
    assert ("Carol", "coauthored_with", "Bob") in paths[0]

    nbrs = neighborhood(["Carol"], hops=1)
    assert "Bob" in nbrs
