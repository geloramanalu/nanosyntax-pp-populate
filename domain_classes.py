import json
from typing import List, Dict, Optional
import nltk

# Load JSON helpers
def load_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Base Node for Nanosyntax tree
class NSNode:
    def __init__(self, label: str, features: Optional[List] = None):
        self.label = label
        self.features = []
        # features here are morphological annotations, not lexicalization cues
        for f in (features or []):
            if f is None:
                continue
            if isinstance(f, str):
                # strip star if present
                value = f.lstrip('*')
            elif isinstance(f, dict):
                # represent morphological dict as 'key:value'
                key = next(iter(f.keys()))
                value = f"{key}:{f[key]}"
            else:
                value = str(f)
            if value:
                self.features.append(value)
        self.children: List[NSNode] = []

    def add_child(self, node: 'NSNode'):
        self.children.append(node)

    def find_nodes(self, label: str) -> List['NSNode']:
        matches = []
        if self.label == label:
            matches.append(self)
        for c in self.children:
            matches.extend(c.find_nodes(label))
        return matches

    def to_nltk_tree(self) -> nltk.Tree:
        if self.label == 'PathP' or self.label == 'PlaceP':
            lbl = self.label
        else:
            lbl = f"{self.label}{{{' '.join(self.features)}}}" if self.features else self.label
        # lbl = self.label
        return nltk.Tree(lbl, [child.to_nltk_tree() for child in self.children])

    def __repr__(self, level=0):
        indent = '  ' * level
        feats = f" {self.features}" if self.features else ''
        s = f"{indent}{self.label}{feats}\n"
        for c in self.children:
            s += c.__repr__(level + 1)
        return s

# Specific syntactic categories based on Svenonius hierarchy
class D(NSNode):
    def __init__(self, features=None):
        super().__init__('D', features)

class K(NSNode):
    def __init__(self, features=None):
        super().__init__('K', features)
        self.add_child(D())

class AxPart(NSNode):
    def __init__(self, features=None):
        super().__init__('AxPart', features)
        self.add_child(K())

class Proj(NSNode):
    def __init__(self, features=None):
        super().__init__('Proj', features)
        self.add_child(AxPart())

class Deg(NSNode):
    def __init__(self, features=None):
        super().__init__('Deg', features)
        self.add_child(Proj())

class PSpatial(NSNode):
    def __init__(self, domain: str, features=None):
        super().__init__(domain, features)
        # Build the standard spine
        if domain == 'PathP':
            self.add_child(Deg())
            self.add_child(NSNode('Path'))
        else:
            self.add_child(Deg())

# Classification function
def classify_pp(head: str,
                path_set: set,
                place_set: set,
                lex: Dict[str, Dict]) -> str:
    if head in path_set:
        return 'PathP'
    if head in place_set:
        return 'PlaceP'
    entry = lex.get(head, {})
    roles = set(entry.get('path_p_morphology', []))
    if roles & {'GOAL', 'SOURCE', 'ROUTE'}:
        return 'PathP'
    return 'PlaceP'

# PP Decomposer class
class PPDecomposer:
    def __init__(self,
                 atomic_path: str,
                 p_lexicon_path: str,
                 complex_path: str,
                 path_set: set,
                 place_set: set):
        self.atomic = load_json(atomic_path)
        self.plex = load_json(p_lexicon_path)
        self.complex = load_json(complex_path)
        self.path_set = path_set
        self.place_set = place_set
        # merge lexicons: atomic overrides p_lex
        self.lex = {**self.plex, **self.atomic}
        self.lex_keys = set(self.lex.keys())

    def _morph_segment(self, token: str) -> List[str]:
        n = len(token)
        # DP table to store valid segmentations from each index
        dp = [None] * (n + 1)
        dp[n] = []  # Base case: empty string has one valid segmentation (empty list)

        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n + 1):
                sub = token[i:j]
                # If we found a morpheme and the rest of the string can also be segmented
                if sub in self.lex_keys and dp[j] is not None:
                    dp[i] = [sub] + dp[j]
                    break  # Found the longest possible morpheme starting at i, move to i-1
        
        # If dp[0] is not None, we found a full segmentation
        if dp[0] is not None:
            return dp[0]
        else:
            # Otherwise, the token cannot be fully segmented. Return it as a single unit.
            return [token]

    def segment(self, pp: str) -> List[str]:
        words = pp.lower().split()
        segments: List[str] = []
        i = 0
        n = len(words)
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
                pieces = self._morph_segment(tok)
                segments.extend(pieces)
                i += 1
        return segments

    def build_tree(self, pp: str) -> NSNode:
        chunks = self.segment(pp)
        head = chunks[-1]
        domain = classify_pp(head, self.path_set, self.place_set, self.lex)
        # morphological annotation for root
        root_feats = self.lex.get(head, {}).get('spellOutHEAD', [])
        root = PSpatial(domain, root_feats)

        for chunk in chunks:
            entry = self.lex.get(chunk, {})
            raw_feats = entry.get('spellOutHEAD', [])
            
            lexical_cues = [f.lstrip('*') for f in raw_feats if isinstance(f, str) and f.startswith('*')]
            morph_feats = [f for f in raw_feats if not (isinstance(f, str) and f.startswith('*'))]
            
            # Create a single node for the chunk with all its morphological features
            node_to_place = NSNode(chunk, morph_feats)
            
            placed = False
            # Attempt to place the node using lexical cues
            if lexical_cues:
                # Use the first cue to find a target attachment point.
                # This prevents attaching the same chunk multiple times.
                target_label = lexical_cues[0]
                target_nodes = root.find_nodes(target_label)
                if target_nodes:
                    # Attach the single, feature-rich node to the first target found
                    target_nodes[0].add_child(node_to_place)
                    placed = True
            
            # If the node was not placed via a cue, attach it to the root as a fallback
            if not placed:
                root.add_child(node_to_place)
                
        return root

    def decompose_all(self) -> Dict[str, nltk.Tree]:
        trees = {}
        # Sorting the list to ensure consistent output for verification
        for pp in sorted(list(self.complex)):
            ns_root = self.build_tree(pp)
            trees[pp] = ns_root.to_nltk_tree()
        return trees



def main():
    PATH_LEMMA_SET = {'to','from','into','onto','through','across','toward','past'}
    PLACE_LEMMA_SET = {'in','on','at','under','beside','near','between','among'}

    decomposer = PPDecomposer(
        'pp_lexicon/atomic_p.json',
        'pp_lexicon/p_lexicon.json',
        'pp_lexicon/complex_pp.json',
        PATH_LEMMA_SET,
        PLACE_LEMMA_SET
    )
    trees = decomposer.decompose_all()
    for pp, tree in trees.items():
        print(f"PP: {pp}")
        # print(tree)
        print(tree.pretty_print())

if __name__ == '__main__':
    main()
