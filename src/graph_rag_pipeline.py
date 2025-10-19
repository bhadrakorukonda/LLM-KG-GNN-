import argparse, json, textwrap

def format_prompt(question, path_texts, k=5):
    context = "\n".join(f"- {t}" for t in path_texts[:k])
    return textwrap.dedent(f'''
    You are a factual assistant. Use ONLY the graph facts below.

    Graph paths:
    {context}

    Question: {question}
    Answer in 1?3 sentences, and mention which path indices you used.
    ''')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths_file", default="edgeaware_paths.jsonl")
    args = ap.parse_args()

    for line in open(args.paths_file, encoding="utf-8-sig"):
        item = json.loads(line)
        prompt = format_prompt(item["q"], [p["text"] for p in item["paths"]])
        print("\n--- PROMPT ---\n")
        print(prompt)
        break  # just show one for now

if __name__ == "__main__":
    main()
