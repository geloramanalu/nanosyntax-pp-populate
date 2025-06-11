import json
from typing import List, Dict, Optional
import nltk
import re

def load_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Semantic classification function
def classify_pp(head: str,
                path_set: set,
                place_set: set,
                lex: Dict[str, Dict]) -> str:
    """
    Classifies a PP as PathP or PlaceP, correctly handling lists and
    null values in path_p_morphology.
    """
    if head in path_set:
        return 'PathP'
    if head in place_set:
        return 'PlaceP'
    
    # Get the morphology entry from the lexicon
    morphology = lex.get(head, {}).get('path_p_morphology', [])
    
    # Standardize the entry to a set for easy processing
    if isinstance(morphology, str):
        roles = {morphology}
    else:
        roles = set(morphology)
        
    # Filter out non-spatial/null values before checking
    roles.discard('none')
    roles.discard(None)
    
    # Check if any valid spatial roles remain
    return 'PathP' if roles & {'GOAL', 'SOURCE', 'ROUTE'} else 'PlaceP'



# Node class for Nanosyntax tree
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

# base class for any Prepositional Phrase
class PPBase:
    lex: Dict[str, Dict]
    lex_keys: set
    path_set: set
    place_set: set
    SPINE = ['p', 'Deg', 'Proj', 'AxPart', 'K', 'D']
    
    # Class-level sorted keys for efficient reuse in segmentation
    sorted_lex_keys: List[str] = []

    @classmethod
    def configure(cls, lexicon: Dict[str, Dict], path_set: set, place_set: set):
        cls.lex = lexicon
        cls.lex_keys = set(lexicon.keys())
        cls.path_set = path_set
        cls.place_set = place_set
        cls.sorted_lex_keys = sorted(list(cls.lex_keys), key=len, reverse=True)

    def __init__(self, raw: str):
        self.raw = raw
        self.tokens = self.segment(raw)
        self.head = self.tokens[-1] if self.tokens else None
        self.domain = classify_pp(self.head, self.path_set, self.place_set, self.lex) if self.head else 'PlaceP'
        self.ns_root = self.build_spine()
        self.attach_lexical_items(self.ns_root)
        self.tree = self.ns_root.to_nltk_tree()

    # --- NEW HELPER METHOD FOR CLASS CHECKING ---
    def is_spatial(self, token: str) -> bool:
        """
        Checks if a token has a spatial class, ignoring NOT_SPATIAL if other
        spatial classes are present.
        """
        SPATIAL_CLASSES = {'PROJECTIVE', 'BOUNDED', 'EXTENDED', 'PARTICLE'}
        entry = self.lex.get(token, {})
        class_info = entry.get('class', 'NOT_SPATIAL')

        if isinstance(class_info, str):
            classes = {class_info}
        else:
            classes = set(class_info)

        return bool(classes & SPATIAL_CLASSES)

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

    # --- REFINED MORPHOLOGICAL SEGMENTATION ---
    def _morph_segment(self, token: str) -> List[str]:
        """
        A recursive segmenter that first tries to match known morpheme prefixes,
        then checks for substrings of larger keys, before falling back.
        """
        if not token:
            return []

        # 1. Try to match a known morpheme from the start of the token
        for key in self.sorted_lex_keys:
            if token.startswith(key):
                remainder = token[len(key):]
                return [key] + self._morph_segment(remainder)
        
        # 2. If no prefix match, check if token is a substring of a larger key
        for key in self.sorted_lex_keys:
            if token in key and len(token) < len(key):
                # e.g., 'cross' is a substring of 'across'. Treat as a valid segment.
                return [token]

        # 3. If no match at all, it's an unknown chunk
        return [token]

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
            for f in raw_feats:
                if isinstance(f, dict):
                    cue = next(iter(f.keys()))
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

    def export_complex_pp(self, outpath: str = None) -> dict:
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

                # --- substring‐to‐lexicon mapping (e.g. 'cross' → 'across') ---
                if entry is None:
                    mapped_key = next((k for k in self.lex.keys() if s in k), None)
                    if mapped_key:
                        entry = self.lex[mapped_key]
                        s = mapped_key  # use the mapped key for heads/roles

                if entry:
                    # collect classes
                    cls = entry.get('class')
                    if cls:
                        classes.extend(cls if isinstance(cls, list) else [cls])

                    # collect raw spellOutHEAD cues
                    for f in entry.get('spellOutHEAD', []):
                        raw_heads.append((s, f))

                    # collect path morphology
                    role = entry.get('path_p_morphology')
                    if role is not None:
                        roles.append(role)

                    # collect measure_allowed
                    if entry.get('measure_allowed'):
                        meas = True
                else:
                    unlex.append(s)

            # --- refine classes: drop 'NOT_SPATIAL' when any other class is present ---
            if classes:
                non_ns = [c for c in classes if c != 'NOT_SPATIAL']
                if non_ns:
                    classes = non_ns

            is_atomic = (len(segs) == 1 and
                         self.lex.get(segs[0], {}).get('isAtomicMorph', False))

            # --- build final spellOutHEAD: mapping overt cues and collecting starred cues ---
            mapped = []
            starred = []
            for chunk, f in raw_heads:
                if isinstance(f, dict):
                    # overt dict cue, keep as-is
                    mapped.append(f)
                elif isinstance(f, str) and not f.startswith('*'):
                    # overt string cue, map to chunk
                    cue = f
                    if cue in SPINE_ORDER:
                        mapped.append({cue: chunk})
                elif isinstance(f, str) and f.startswith('*'):
                    # non-overt cue
                    starred.append(f)

            # merge and dedupe
            final_heads = mapped + starred
            seen = set()
            uniq_heads = []
            for h in final_heads:
                key = json.dumps(h) if isinstance(h, dict) else h
                if key not in seen:
                    seen.add(key)
                    uniq_heads.append(h)

            # sort in reverse SPINE order
            def head_index(h):
                if isinstance(h, dict):
                    cue = next(iter(h.keys()))
                else:
                    cue = h.lstrip('*')
                return SPINE_ORDER.index(cue) if cue in SPINE_ORDER else len(SPINE_ORDER)
            uniq_heads.sort(key=head_index, reverse=True)

            # --- refine spellOutHEAD: remove starred cues that duplicate overt cues ---
            filtered_heads = []
            seen_cues = set()
            for h in uniq_heads:
                if isinstance(h, dict):
                    cue = next(iter(h.keys()))
                    seen_cues.add(cue)
                    filtered_heads.append(h)
                else:
                    cue = h.lstrip('*')
                    if cue in seen_cues:
                        continue
                    filtered_heads.append(h)
                    seen_cues.add(cue)
            uniq_heads = filtered_heads

            # --- flatten and dedupe path_p_morphology ---
            flat_roles = []
            for r in roles:
                if isinstance(r, list):
                    for v in r:
                        if v not in flat_roles:
                            flat_roles.append(v)
                else:
                    if r not in flat_roles:
                        flat_roles.append(r)

            # --- refine path_p_morphology: drop 'none' when any other role is present ---
            if flat_roles:
                non_none = [r for r in flat_roles if r != 'none']
                if non_none:
                    flat_roles = non_none

            # build the final entry
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

        if outpath:
            with open(outpath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

        return result

