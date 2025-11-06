from __future__ import annotations
import json, re, difflib, yaml, os
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?")

def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

def token_windows(words: List[str], min_n=1, max_n=6) -> List[Tuple[int,int,str]]:
    out = []
    L = len(words)
    for n in range(min_n, min(max_n, L) + 1):
        for i in range(0, L - n + 1):
            span = " ".join(words[i:i+n])
            out.append((i, i+n, span))
    return out

class AliasIndex:
    def __init__(self):
        self.alias2ids: Dict[str, Set[str]] = defaultdict(set)
        self.first_token_map: Dict[str, Set[str]] = defaultdict(set)
        self.max_alias_len_words = 1

    def add(self, canonical_id: str, alias: str):
        a = normalize(alias)
        if not a: return
        self.alias2ids[a].add(canonical_id)
        first = a.split(" ", 1)[0]
        self.first_token_map[first].add(a)
        self.max_alias_len_words = max(self.max_alias_len_words, len(a.split()))

    def build_from_jsonl(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                cid = rec["id"]
                for al in rec.get("aliases", []):
                    self.add(cid, al)

    def lookup_exact(self, phrase: str) -> Set[str]:
        return self.alias2ids.get(normalize(phrase), set())

    def fuzzy_candidates_for_first(self, first_token: str) -> Set[str]:
        return self.first_token_map.get(first_token, set())

class RelationLexicon:
    def __init__(self):
        self.phrase2rel: Dict[str, str] = {}
        self.rel_meta: Dict[str, Dict] = {}
        self.max_phrase_len_words = 1

    def load_schema(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            schema = yaml.safe_load(f)
        for rel in schema.get("relations", []):
            rid = rel["id"]
            phrases = rel.get("phrases", [])
            self.rel_meta[rid] = {
                "symmetric": bool(rel.get("symmetric", False)),
                "inverse_id": rel.get("inverse_id")
            }
            for p in phrases:
                pn = normalize(p)
                self.phrase2rel[pn] = rid
                self.max_phrase_len_words = max(self.max_phrase_len_words, len(pn.split()))
        return self

    def lookup_exact(self, phrase: str) -> Optional[str]:
        return self.phrase2rel.get(normalize(phrase))

    def all_phrases_starting_with(self, first_token: str) -> List[Tuple[str,str]]:
        first = normalize(first_token)
        out = []
        for p, rid in self.phrase2rel.items():
            if p.split(" ", 1)[0] == first:
                out.append((p, rid))
        return out

class QuestionLinker:
    def __init__(self, aliases_path: str, schema_path: str, fuzzy: bool = True):
        if not os.path.exists(aliases_path):
            raise FileNotFoundError(f"aliases.jsonl not found: {aliases_path}")
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"schema.yaml not found: {schema_path}")

        self.alias_index = AliasIndex()
        self.alias_index.build_from_jsonl(aliases_path)

        self.rel_lex = RelationLexicon().load_schema(schema_path)
        self.enable_fuzzy = fuzzy

        self.entity_ratio_thresh = 90 if not HAVE_RAPIDFUZZ else 88
        self.relation_ratio_thresh = 90 if not HAVE_RAPIDFUZZ else 88

    def _similar(self, a: str, b: str, rel=False) -> int:
        if HAVE_RAPIDFUZZ:
            return fuzz.token_set_ratio(a, b)
        return int(difflib.SequenceMatcher(a=normalize(a), b=normalize(b)).ratio() * 100)

    def link(self, question: str, max_entities: int = 5):
        qnorm = normalize(question)
        words = _WORD_RE.findall(qnorm)
        if not words:
            return [], set()

        seed_ids: List[str] = []
        seen_ids: Set[str] = set()

        max_ent_len = min(self.alias_index.max_alias_len_words, 6)
        spans = token_windows(words, 1, max_ent_len)

        for (_, _, span) in spans:
            first = span.split(" ", 1)[0]
            candidate_aliases = self.alias_index.fuzzy_candidates_for_first(first)
            exact_ids = self.alias_index.lookup_exact(span)
            if exact_ids:
                for cid in exact_ids:
                    if cid not in seen_ids:
                        seed_ids.append(cid); seen_ids.add(cid)
                if len(seed_ids) >= max_entities: break
                continue
            if self.enable_fuzzy and candidate_aliases:
                best_alias = None; best_score = 0
                for al in candidate_aliases:
                    score = self._similar(span, al)
                    if score > best_score:
                        best_alias, best_score = al, score
                if best_alias and best_score >= self.entity_ratio_thresh:
                    for cid in self.alias_index.alias2ids[best_alias]:
                        if cid not in seen_ids:
                            seed_ids.append(cid); seen_ids.add(cid)
                if len(seed_ids) >= max_entities: break

        seed_ids = list(dict.fromkeys(seed_ids))

        rel_hints: Set[str] = set()
        max_rel_len = min(self.rel_lex.max_phrase_len_words, 6)
        spans_rel = token_windows(words, 1, max_rel_len)

        for (_, _, span) in spans_rel:
            rid = self.rel_lex.lookup_exact(span)
            if rid:
                rel_hints.add(rid)
                continue
            if self.enable_fuzzy:
                first = span.split(" ", 1)[0]
                candidates = self.rel_lex.all_phrases_starting_with(first)
                if not candidates: continue
                best_rid, best_score = None, 0
                for p, r in candidates:
                    score = self._similar(span, p, rel=True)
                    if score > best_score:
                        best_rid, best_score = r, score
                if best_rid and best_score >= self.relation_ratio_thresh:
                    rel_hints.add(best_rid)

        return seed_ids, rel_hints
