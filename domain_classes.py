import json
from typing import List, Dict, Optional
import nltk
import re

# Load JSON helpers
def load_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Semantic classification function
def classify_pp(head: str,
                path_set: set,
                place_set: set,
                lex: Dict[str, Dict]) -> str:
    if head in path_set:
        return 'PathP'
    if head in place_set:
        return 'PlaceP'
    roles = set(lex.get(head, {}).get('path_p_morphology', []))
    return 'PathP' if roles & {'GOAL','SOURCE','ROUTE'} else 'PlaceP'

# Node class for Nanosyntax tree
typing_NSNode = None  # forward
class NSNode:
    def __init__(self, label: str, features: Optional[List] = None):
        self.label = label
        self.features: List[str] = []
        self.children: List['NSNode'] = []
        for f in (features or []):
            if isinstance(f, str):
                # preserve '*' for non-overt cues
                self.features.append(f)
            elif isinstance(f, dict):
                # dict keys are overt cues
                self.features.append(next(iter(f.keys())))

    def add_child(self, node: 'NSNode'):
        self.children.append(node)

    def find_nodes(self, label: str) -> List['NSNode']:
        result = []
        if self.label == label:
            result.append(self)
        for c in self.children:
            result.extend(c.find_nodes(label))
        return result

    def to_nltk_tree(self) -> nltk.Tree:
        feat_str = ' '.join(self.features)
        lbl = f"{self.label}{{{feat_str}}}" if self.features else self.label
        return nltk.Tree(lbl, [c.to_nltk_tree() for c in self.children])

    def __repr__(self, level=0):
        indent = '  ' * level
        feat_str = f" [{','.join(self.features)}]" if self.features else ''
        s = f"{indent}{self.label}{feat_str}\n"
        for c in self.children:
            s += c.__repr__(level+1)
        return s

# Abstract base class for any Prepositional Phrase
class PPBase:
    lex: Dict[str, Dict]
    lex_keys: set
    path_set: set
    place_set: set
    SPINE = ['p', 'Deg', 'Proj', 'AxPart', 'K', 'D']

    @classmethod
    def configure(cls, lexicon: Dict[str, Dict], path_set: set, place_set: set):
        cls.lex = lexicon
        cls.lex_keys = set(lexicon.keys())
        cls.path_set = path_set
        cls.place_set = place_set

    def __init__(self, raw: str):
        self.raw = raw
        self.tokens = self.segment(raw)
        self.head = self.tokens[-1] if self.tokens else None
        self.domain = classify_pp(self.head, self.path_set, self.place_set, self.lex) if self.head else 'PlaceP'
        self.ns_root = self.build_spine()
        self.attach_lexical_items(self.ns_root)
        self.tree = self.ns_root.to_nltk_tree()

    def segment(self, pp: str) -> List[str]:
        words = pp.lower().split()
        segments, i, n = [], 0, len(words)
        while i < n:
            match = None
            for j in range(n, i, -1):
                span = ' '.join(words[i:j])
                if span in self.lex_keys:
                    match = span
                    break
            if match:
                segments.append(match)
                i += len(match.split())
            else:
                tok = words[i]
                segments.extend(self._morph_segment(tok))
                i += 1
        return segments

    def _morph_segment(self, token: str) -> List[str]:
        segs, i, n = [], 0, len(token)
        while i < n:
            sub = None
            for j in range(n, i, -1):
                part = token[i:j]
                if part in self.lex_keys:
                    sub = part
                    break
            if sub:
                segs.append(sub)
                i += len(sub)
            else:
                segs.append(token[i:])
                break
        return segs

    def build_spine(self) -> NSNode:
        root = NSNode(self.domain)
        current = root
        if self.domain == 'PathP':
            path_node = NSNode('Path')
            current.add_child(path_node)
            current = path_node
        for label in self.SPINE:
            node = NSNode(label)
            current.add_child(node)
            current = node
        return root

    def attach_lexical_items(self, spine: NSNode):
        for chunk in self.tokens:
            entry = self.lex.get(chunk, {})
            raw_feats = entry.get('spellOutHEAD', [])
            node = NSNode(chunk, raw_feats)
            placed = False
            # attach under overt cues (dict keys) and unstarred string cues
            for f in raw_feats:
                if isinstance(f, dict):
                    cue = next(iter(f.keys()))
                elif isinstance(f, str) and not f.startswith('*'):
                    cue = f
                else:
                    continue
                for target in spine.find_nodes(cue):
                    target.add_child(node)
                    placed = True
            if not placed:
                spine.add_child(node)
# Helper to generate valid Python class names for each PP
def _create_class_name(pp: str) -> str:
    parts = re.split(r"\W+", pp)
    return 'PP_' + ''.join(p.title() for p in parts if p)

# Factory to create PP subclasses dynamically
class PPFactory:
    def __init__(self, atomic_path: str, p_lexicon_path: str, complex_path: str,
                 path_set: set, place_set: set):
        atomic = load_json(atomic_path)
        plex  = load_json(p_lexicon_path)
        lex = {**plex, **atomic}
        PPBase.configure(lex, path_set, place_set)
        self.pp_list = load_json(complex_path)
        self.classes = {}

    def create_classes(self) -> Dict[str, type]:
        for pp in self.pp_list:
            name = _create_class_name(pp)
            NewClass = type(name, (PPBase,), {})
            self.classes[pp] = NewClass
        return self.classes

    def instantiate_all(self) -> Dict[str, PPBase]:
        return {pp: Cls(pp) for pp, Cls in self.classes.items()}

# Extend PPFactory to export decomposed lexicon
class PPFactory:
    def __init__(self, atomic_path: str, p_lexicon_path: str, complex_path: str,
                 path_set: set, place_set: set):
        atomic = load_json(atomic_path)
        plex  = load_json(p_lexicon_path)
        self.lex = {**plex, **atomic}
        PPBase.configure(self.lex, path_set, place_set)
        self.pp_list = load_json(complex_path)
        self.classes = {}

    def create_classes(self) -> Dict[str, type]:
        for pp in self.pp_list:
            name = _create_class_name(pp)
            NewClass = type(name, (PPBase,), {})
            self.classes[pp] = NewClass
        return self.classes

    def instantiate_all(self) -> Dict[str, PPBase]:
        return {pp: Cls(pp) for pp, Cls in self.classes.items()}

    def export_complex_pp(self, outpath: str):
        # Build entries for each PP
        instances = self.instantiate_all()
        result = {}
        SPINE_ORDER = ['Path','p','Proj','Deg','AxPart','K','D']
        for pp, obj in instances.items():
            segs = obj.tokens
            classes = []
            raw_heads = []
            roles = []
            meas = False
            unlex = []
            for s in segs:
                entry = self.lex.get(s)
                if entry:
                    cls = entry.get('class')
                    if cls:
                        classes.extend(cls if isinstance(cls, list) else [cls])
                    for f in entry.get('spellOutHEAD', []):
                        raw_heads.append((s, f))
                    role = entry.get('path_p_morphology')
                    if role is not None:
                        roles.append(role)
                    if entry.get('measure_allowed'):
                        meas = True
                else:
                    unlex.append(s)
            is_atomic = len(segs)==1 and self.lex.get(segs[0],{}).get('isAtomicMorph',False)
            # build final spellOutHEAD
            final_heads = []
            for chunk, f in raw_heads:
                if isinstance(f, dict):
                    final_heads.append(f)
                elif isinstance(f, str) and not f.startswith('*'):
                    final_heads.append({f: chunk})
                elif isinstance(f, str) and f.startswith('*'):
                    final_heads.append(f)
            seen = set(); uniq_heads = []
            for h in final_heads:
                key = json.dumps(h) if isinstance(h, dict) else h
                if key not in seen:
                    seen.add(key); uniq_heads.append(h)
            uniq_heads.sort(key=lambda h: SPINE_ORDER.index(next(iter(h.keys())) if isinstance(h, dict) else h.lstrip('*')), reverse=True)
            flat_roles = []
            for r in roles:
                if isinstance(r, list):
                    for v in r:
                        if v not in flat_roles:
                            flat_roles.append(v)
                else:
                    if r not in flat_roles:
                        flat_roles.append(r)
            entry = {
                'isAtomicMorph': is_atomic,
                'class': classes or None,
                'spellOutHEAD': uniq_heads,
                'path_p_morphology': flat_roles or None,
                'measure_allowed': meas
            }
            if unlex:
                entry['unlexicalized'] = unlex
            result[pp] = entry
        with open(outpath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

# Example usage
if __name__=='__main__':
    PATH={'to','from','into','onto','through','across','toward','past'}
    PLACE={'in','on','at','under','beside','near','between','among'}
    factory = PPFactory('pp_lexicon/atomic_p.json', 'pp_lexicon/p_lexicon.json', 'pp_lexicon/complex_pp.json', PATH, PLACE)
    factory.create_classes()
    # Export the augmented lexicon
    factory.export_complex_pp('complex_pp_expanded.json')
    print("Wrote complex_pp_expanded.json")
