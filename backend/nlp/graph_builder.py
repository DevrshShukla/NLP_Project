from typing import List, Dict, Optional, Tuple
from backend.nlp.topic_ranker import ScoredTopic
from backend.models.schemas import RelationshipGraph, GraphNode, GraphEdge

class GraphBuilder:
    @classmethod
    def build_graph(
        cls,
        topics: List[ScoredTopic],
        parent_map: Dict[str, Optional[str]],
        depth_map: Dict[str, int],
        sentences: List[str],
        max_nodes: int = 40,
        max_edges: int = 120
    ) -> RelationshipGraph:
        # Cap to top max_nodes topics
        top_topics = topics[:max_nodes]
        nodes: List[GraphNode] = []
        topic_keys = set()

        for t in top_topics:
            k = t.candidate.key
            topic_keys.add(k)
            nodes.append(GraphNode(
                id=k,
                label=t.candidate.surface,
                score=t.score,
                priority=t.priority,
                cluster_id=None,
                depth=depth_map.get(k, 0)
            ))

        edges: List[GraphEdge] = []
        edge_keys = set()

        # 1. Add hierarchy edges
        for k in topic_keys:
            parent_k = parent_map.get(k)
            if parent_k and parent_k in topic_keys:
                edge_id = (parent_k, k, "hierarchy")
                if edge_id not in edge_keys:
                    edge_keys.add(edge_id)
                    edges.append(GraphEdge(
                        source=parent_k,
                        target=k,
                        type="hierarchy",
                        weight=0.85
                    ))

        # 2. Add co-occurrence / semantic edges from sentences
        # Count pairwise co-occurrence in sentences
        co_counts: Dict[Tuple[str, str], int] = {}
        for sent in sentences:
            s_lower = sent.lower()
            matching_keys = [k for k in topic_keys if k in s_lower]
            for i in range(len(matching_keys)):
                for j in range(i + 1, len(matching_keys)):
                    a, b = matching_keys[i], matching_keys[j]
                    pair = (min(a, b), max(a, b))
                    co_counts[pair] = co_counts.get(pair, 0) + 1

        # Sort pairs by co-occurrence count DESC
        sorted_pairs = sorted(co_counts.items(), key=lambda item: -item[1])
        max_co = sorted_pairs[0][1] if sorted_pairs else 1

        for (a, b), count in sorted_pairs:
            if len(edges) >= max_edges:
                break
            if count >= 1:
                # Normalized weight 0.3 to 0.95
                norm_w = round(0.3 + 0.65 * (count / max_co), 2)
                edge_id = (a, b, "semantic")
                if edge_id not in edge_keys:
                    edge_keys.add(edge_id)
                    edges.append(GraphEdge(
                        source=a,
                        target=b,
                        type="semantic",
                        weight=norm_w
                    ))

        return RelationshipGraph(nodes=nodes, edges=edges)
