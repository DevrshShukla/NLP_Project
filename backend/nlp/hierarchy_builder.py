import re
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
from backend.nlp.topic_ranker import ScoredTopic
from backend.models.schemas import HierarchyNode

class HierarchyBuilder:
    PATTERN_INCLUDES = re.compile(
        r'\b(includes?|such\s+as|consists?\s+of|types?\s+of|kinds?\s+of|comprises?|categories?\s+of)\b',
        re.IGNORECASE
    )

    @classmethod
    def compute_evidence(
        cls,
        parent: ScoredTopic,
        child: ScoredTopic,
        sentences: List[str],
        total_sections: int
    ) -> float:
        if parent.candidate.key == child.candidate.key:
            return 0.0

        p_words = set(parent.candidate.key.split())
        c_words = set(child.candidate.key.split())

        # Lexical containment: tokens(p) is proper subset of tokens(c)
        lex = 0.0
        if p_words.issubset(c_words) and len(p_words) < len(c_words):
            lex = 1.0
        elif parent.candidate.key.split()[-1] == child.candidate.key.split()[-1]:
            # Same head noun
            p_gen = len(parent.candidate.section_ids) / max(1, total_sections)
            c_gen = len(child.candidate.section_ids) / max(1, total_sections)
            if p_gen > c_gen or parent.score > child.score:
                lex = 0.5

        # Pattern match: "p (includes|such as|...) c"
        pat = 0.0
        p_name = parent.candidate.surface.lower()
        c_name = child.candidate.surface.lower()
        
        for sent in sentences:
            s_lower = sent.lower()
            if p_name in s_lower and c_name in s_lower:
                match = cls.PATTERN_INCLUDES.search(s_lower)
                if match:
                    p_pos = s_lower.find(p_name)
                    c_pos = s_lower.find(c_name)
                    if p_pos < match.start() < c_pos:
                        pat = 1.0
                        break

        # Semantic cosine similarity (if available, otherwise estimate via jaccard / co-occurrence)
        sem = 0.0
        shared_sections = len(parent.candidate.section_ids.intersection(child.candidate.section_ids))
        if shared_sections > 0:
            sem = min(1.0, shared_sections / max(1, len(parent.candidate.section_ids)))

        evidence = 0.40 * lex + 0.30 * pat + 0.30 * sem
        return round(evidence, 4)

    @classmethod
    def build_hierarchy(
        cls,
        topics: List[ScoredTopic],
        sentences: List[str],
        total_sections: int,
        threshold: float = 0.35
    ) -> Tuple[Dict[str, Optional[str]], Dict[str, List[str]], Dict[str, int], List[HierarchyNode]]:
        """
        Builds a directed tree/forest with max depth 4 using NetworkX.
        Returns:
            parent_map: topic_id -> parent_topic_id
            children_map: topic_id -> [child_topic_ids]
            depth_map: topic_id -> depth (0..4)
            hierarchy_roots: List[HierarchyNode] tree structure for UI
        """
        parent_map: Dict[str, Optional[str]] = {t.candidate.key: None for t in topics}
        children_map: Dict[str, List[str]] = {t.candidate.key: [] for t in topics}
        depth_map: Dict[str, int] = {t.candidate.key: 0 for t in topics}

        if len(topics) < 2:
            roots = [
                HierarchyNode(
                    id=t.candidate.key,
                    name=t.candidate.surface,
                    score=t.score,
                    priority=t.priority,
                    depth=0,
                    children=[]
                ) for t in topics
            ]
            return parent_map, children_map, depth_map, roots

        G = nx.DiGraph()
        for t in topics:
            G.add_node(t.candidate.key, topic=t)

        # For each topic c, evaluate candidate parents
        for child in topics:
            best_parent = None
            best_evidence = 0.0
            for potential_parent in topics:
                if potential_parent.candidate.key == child.candidate.key:
                    continue
                # Parent must generally have score >= child or more sections
                if potential_parent.score < child.score and len(potential_parent.candidate.section_ids) < len(child.candidate.section_ids):
                    continue
                    
                ev = cls.compute_evidence(potential_parent, child, sentences, total_sections)
                if ev >= threshold and ev > best_evidence:
                    best_evidence = ev
                    best_parent = potential_parent

            if best_parent and best_evidence >= threshold:
                G.add_edge(best_parent.candidate.key, child.candidate.key, weight=best_evidence)

        # Cycle resolution: remove lowest-weight edge in cycle until acyclic
        while True:
            try:
                cycle = nx.find_cycle(G, orientation="original")
                # Find edge with lowest weight in cycle
                min_edge = min(cycle, key=lambda edge: G[edge[0]][edge[1]].get("weight", 0.0))
                G.remove_edge(min_edge[0], min_edge[1])
            except nx.NetworkXNoCycle:
                break

        # Ensure each node has at most one parent (in-degree <= 1)
        for node in list(G.nodes()):
            in_edges = list(G.in_edges(node, data=True))
            if len(in_edges) > 1:
                # Keep only the edge with highest weight
                best_in = max(in_edges, key=lambda e: e[2].get("weight", 0.0))
                for edge in in_edges:
                    if edge != best_in:
                        G.remove_edge(edge[0], edge[1])

        # Compute depths and clamp to 4
        roots_keys = [n for n in G.nodes() if G.in_degree(n) == 0]
        
        for root in roots_keys:
            # BFS or DFS traversal to assign depths
            queue = [(root, 0, None)]
            while queue:
                curr, d, p = queue.pop(0)
                if d > 4:
                    d = 4  # Clamped
                depth_map[curr] = d
                parent_map[curr] = p
                
                children = [c for _, c in G.out_edges(curr)]
                children_map[curr] = children
                for c in children:
                    queue.append((c, d + 1, curr))

        # Build hierarchy tree nodes
        topic_dict = {t.candidate.key: t for t in topics}

        def build_node(k: str) -> HierarchyNode:
            t = topic_dict[k]
            child_nodes = [build_node(c) for c in children_map.get(k, [])]
            return HierarchyNode(
                id=k,
                name=t.candidate.surface,
                score=t.score,
                priority=t.priority,
                depth=depth_map.get(k, 0),
                children=child_nodes
            )

        hierarchy_roots = [build_node(r) for r in roots_keys]
        return parent_map, children_map, depth_map, hierarchy_roots
