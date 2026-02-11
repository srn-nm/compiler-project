"""
Graph similarity algorithms for CFG comparison
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Any
from collections import deque
from .cfg_builder import ControlFlowGraph, CFGNode, NodeType


class GraphSimilarity:
    """Graph similarity algorithms for CFG comparison"""

    def __init__(self, weight_structural: float = 0.4,
                 weight_graph_edit: float = 0.3,
                 weight_subgraph: float = 0.3):
        self.weights = {
            'structural': weight_structural,
            'graph_edit': weight_graph_edit,
            'subgraph': weight_subgraph
        }

    def calculate_graph_similarity(self, cfg1: ControlFlowGraph,
                                   cfg2: ControlFlowGraph) -> Dict[str, Any]:
        """Calculate comprehensive graph similarity"""
        metrics = {}

        structural_sim = self.structural_similarity(cfg1, cfg2)
        metrics['structural_similarity'] = structural_sim

        graph_edit_sim = self.graph_edit_distance_similarity(cfg1, cfg2)
        metrics['graph_edit_similarity'] = graph_edit_sim

        subgraph_sim = self.subgraph_matching_similarity(cfg1, cfg2)
        metrics['subgraph_similarity'] = subgraph_sim

        path_sim = self.execution_path_similarity(cfg1, cfg2)
        metrics['path_similarity'] = path_sim

        node_type_sim = self.node_type_distribution_similarity(cfg1, cfg2)
        metrics['node_type_similarity'] = node_type_sim

        overall_score = (
                                self.weights['structural'] * structural_sim +
                                self.weights['graph_edit'] * graph_edit_sim +
                                self.weights['subgraph'] * subgraph_sim
                        ) * 100

        similar_components = self.find_similar_components(cfg1, cfg2)

        return {
            'overall_similarity': overall_score,
            'detailed_metrics': metrics,
            'similar_components': similar_components,
            'weights_applied': self.weights
        }

    def structural_similarity(self, cfg1: ControlFlowGraph,
                              cfg2: ControlFlowGraph) -> float:
        """Structural similarity based on graph features"""
        features1 = self._extract_structural_features(cfg1)
        features2 = self._extract_structural_features(cfg2)

        similarity = 0
        total_features = 0

        for key in features1:
            if key in features2:
                val1 = features1[key]
                val2 = features2[key]

                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if val1 == 0 and val2 == 0:
                        sim = 1.0
                    else:
                        sim = 1 - abs(val1 - val2) / max(val1, val2)
                    similarity += sim
                    total_features += 1
                elif isinstance(val1, dict) and isinstance(val2, dict):
                    sim = self._compare_dictionaries(val1, val2)
                    similarity += sim
                    total_features += 1

        return similarity / total_features if total_features > 0 else 0.0

    def graph_edit_distance_similarity(self, cfg1: ControlFlowGraph,
                                       cfg2: ControlFlowGraph) -> float:
        """Similarity based on graph edit distance"""
        n1 = len(cfg1.nodes)
        n2 = len(cfg2.nodes)

        if n1 == 0 and n2 == 0:
            return 1.0

        nodes1 = list(cfg1.nodes.values())
        nodes2 = list(cfg2.nodes.values())
        similarity_matrix = np.zeros((n1, n2))

        for i, node1 in enumerate(nodes1):
            for j, node2 in enumerate(nodes2):
                similarity_matrix[i, j] = self._node_similarity(node1, node2)

        matched_pairs = self._find_optimal_matching(similarity_matrix)

        edit_cost = 0
        matched_nodes1 = set(i for i, _ in matched_pairs)
        matched_nodes2 = set(j for _, j in matched_pairs)

        edit_cost += (n1 - len(matched_nodes1))
        edit_cost += (n2 - len(matched_nodes2))

        for i, j in matched_pairs:
            edit_cost += 1 - similarity_matrix[i, j]

        edge_cost = self._calculate_edge_edit_cost(cfg1, cfg2, matched_pairs)
        edit_cost += edge_cost

        max_cost = (n1 + n2) + (len(cfg1.edges) + len(cfg2.edges))
        similarity = 1 - (edit_cost / max_cost) if max_cost > 0 else 1.0

        return max(similarity, 0.0)

    def subgraph_matching_similarity(self, cfg1: ControlFlowGraph,
                                     cfg2: ControlFlowGraph) -> float:
        """Similarity based on subgraph matching"""
        # Extract important subgraphs from both CFGs
        subgraphs1 = self._extract_important_subgraphs(cfg1)
        subgraphs2 = self._extract_important_subgraphs(cfg2)

        if not subgraphs1 and not subgraphs2:
            return 1.0

        matched_count = 0
        for sg1 in subgraphs1:
            best_match_score = 0
            for sg2 in subgraphs2:
                match_score = self._subgraph_match_score(sg1, sg2)
                if match_score > best_match_score:
                    best_match_score = match_score

            if best_match_score >= 0.7:
                matched_count += 1

        total_subgraphs = max(len(subgraphs1), len(subgraphs2))
        return matched_count / total_subgraphs if total_subgraphs > 0 else 0.0

    def execution_path_similarity(self, cfg1: ControlFlowGraph,
                                  cfg2: ControlFlowGraph) -> float:
        """Similarity based on execution paths"""
        paths1 = cfg1.get_execution_paths(max_paths=50)
        paths2 = cfg2.get_execution_paths(max_paths=50)

        if not paths1 and not paths2:
            return 1.0

        total_similarity = 0
        comparisons = 0

        for path1 in paths1[:10]:
            best_similarity = 0
            for path2 in paths2[:10]:
                similarity = self._path_similarity(path1, path2)
                if similarity > best_similarity:
                    best_similarity = similarity

            total_similarity += best_similarity
            comparisons += 1

        return total_similarity / comparisons if comparisons > 0 else 0.0

    def node_type_distribution_similarity(self, cfg1: ControlFlowGraph,
                                          cfg2: ControlFlowGraph) -> float:
        """Similarity based on node type distribution"""
        type_dist1 = self._get_node_type_distribution(cfg1)
        type_dist2 = self._get_node_type_distribution(cfg2)

        all_types = set(list(type_dist1.keys()) + list(type_dist2.keys()))

        if not all_types:
            return 1.0

        vec1 = [type_dist1.get(t, 0) for t in all_types]
        vec2 = [type_dist2.get(t, 0) for t in all_types]

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = np.sqrt(sum(a * a for a in vec1))
        norm2 = np.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def find_similar_components(self, cfg1: ControlFlowGraph,
                                cfg2: ControlFlowGraph) -> List[Dict[str, Any]]:
        """Find and return similar components between two CFGs"""
        similar_components = []

        decisions1 = self._get_nodes_by_type(cfg1, NodeType.DECISION)
        decisions2 = self._get_nodes_by_type(cfg2, NodeType.DECISION)

        for d1 in decisions1:
            for d2 in decisions2:
                similarity = self._decision_structure_similarity(cfg1, cfg2, d1, d2)
                if similarity > 0.7:
                    similar_components.append({
                        'type': 'decision_structure',
                        'node1': d1.id,
                        'node2': d2.id,
                        'similarity': similarity
                    })

        loops1 = self._get_nodes_by_type(cfg1, NodeType.LOOP)
        loops2 = self._get_nodes_by_type(cfg2, NodeType.LOOP)

        for l1 in loops1:
            for l2 in loops2:
                similarity = self._loop_structure_similarity(cfg1, cfg2, l1, l2)
                if similarity > 0.7:
                    similar_components.append({
                        'type': 'loop_structure',
                        'node1': l1.id,
                        'node2': l2.id,
                        'similarity': similarity
                    })

        funcs1 = self._get_nodes_by_type(cfg1, NodeType.FUNCTION)
        funcs2 = self._get_nodes_by_type(cfg2, NodeType.FUNCTION)

        for f1 in funcs1:
            for f2 in funcs2:
                similarity = self._function_structure_similarity(cfg1, cfg2, f1, f2)
                if similarity > 0.7:
                    similar_components.append({
                        'type': 'function_structure',
                        'node1': f1.id,
                        'node2': f2.id,
                        'similarity': similarity
                    })

        return similar_components

    def _extract_structural_features(self, cfg: ControlFlowGraph) -> Dict[str, Any]:
        """Extract structural features from CFG"""
        nodes = cfg.nodes

        in_degrees = [len(node.in_edges) for node in nodes.values()]
        out_degrees = [len(node.out_edges) for node in nodes.values()]

        type_dist = {}
        for node in nodes.values():
            node_type = node.type.value
            type_dist[node_type] = type_dist.get(node_type, 0) + 1

        cyclomatic = cfg.get_cyclomatic_complexity()

        return {
            'node_count': len(nodes),
            'edge_count': len(cfg.edges),
            'avg_in_degree': np.mean(in_degrees) if in_degrees else 0,
            'avg_out_degree': np.mean(out_degrees) if out_degrees else 0,
            'max_in_degree': max(in_degrees) if in_degrees else 0,
            'max_out_degree': max(out_degrees) if out_degrees else 0,
            'type_distribution': type_dist,
            'cyclomatic_complexity': cyclomatic,
            'entry_node': cfg.entry_node_id is not None,
            'exit_node': cfg.exit_node_id is not None
        }

    def _node_similarity(self, node1: CFGNode, node2: CFGNode) -> float:
        """Calculate similarity between two nodes"""
        similarity = 0.0

        if node1.type == node2.type:
            similarity += 0.4

        in_degree_sim = 1 - abs(len(node1.in_edges) - len(node2.in_edges)) / \
                        max(len(node1.in_edges), len(node2.in_edges), 1)
        out_degree_sim = 1 - abs(len(node1.out_edges) - len(node2.out_edges)) / \
                         max(len(node1.out_edges), len(node2.out_edges), 1)

        similarity += 0.2 * in_degree_sim
        similarity += 0.2 * out_degree_sim

        if node1.label and node2.label:
            label1 = node1.label.lower()
            label2 = node2.label.lower()

            if label1 == label2:
                similarity += 0.2
            elif any(keyword in label1 and keyword in label2
                     for keyword in ['if', 'while', 'for', 'return', 'call']):
                similarity += 0.1

        return min(similarity, 1.0)

    def _find_optimal_matching(self, similarity_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Find optimal matching between nodes using greedy algorithm"""
        n1, n2 = similarity_matrix.shape
        matched_pairs = []
        threshold = 0.5

        potential_matches = []
        for i in range(n1):
            for j in range(n2):
                if similarity_matrix[i, j] >= threshold:
                    potential_matches.append((similarity_matrix[i, j], i, j))

        potential_matches.sort(reverse=True, key=lambda x: x[0])

        matched_i = set()
        matched_j = set()

        for sim, i, j in potential_matches:
            if i not in matched_i and j not in matched_j:
                matched_pairs.append((i, j))
                matched_i.add(i)
                matched_j.add(j)

        return matched_pairs

    def _calculate_edge_edit_cost(self, cfg1: ControlFlowGraph,
                                  cfg2: ControlFlowGraph,
                                  matched_pairs: List[Tuple[int, int]]) -> float:
        """Calculate edge edit cost based on matched nodes"""
        nodes1 = list(cfg1.nodes.values())
        nodes2 = list(cfg2.nodes.values())

        edge_cost = 0

        mapping = {}
        for i, j in matched_pairs:
            mapping[nodes1[i].id] = nodes2[j].id

        for from_i, to_j, _ in cfg1.edges:
            if from_i in mapping and to_j in mapping:
                edge_exists = False
                for from_id2, to_id2, _ in cfg2.edges:
                    if from_id2 == mapping[from_i] and to_id2 == mapping[to_j]:
                        edge_exists = True
                        break

                if not edge_exists:
                    edge_cost += 1

        return edge_cost * 0.5

    def _extract_important_subgraphs(self, cfg: ControlFlowGraph) -> List[Dict[str, Any]]:
        """Extract important subgraphs from CFG"""
        subgraphs = []

        for node_id, node in cfg.nodes.items():
            if node.type in [NodeType.DECISION, NodeType.LOOP, NodeType.FUNCTION]:
                subgraph = self._extract_neighborhood(cfg, node_id, radius=2)
                if subgraph:
                    subgraphs.append(subgraph)

        return subgraphs

    def _extract_neighborhood(self, cfg: ControlFlowGraph,
                              center_id: int, radius: int = 2) -> Dict[str, Any]:
        """Extract neighborhood around a node"""
        nodes = cfg.nodes
        visited = set()
        queue = deque([(center_id, 0)])
        subgraph_nodes = {}
        subgraph_edges = []

        while queue:
            node_id, distance = queue.popleft()

            if node_id in visited or distance > radius:
                continue

            visited.add(node_id)
            subgraph_nodes[node_id] = nodes[node_id]

            for succ_id in nodes[node_id].out_edges:
                if succ_id in nodes:
                    subgraph_edges.append((node_id, succ_id))
                    if succ_id not in visited and distance + 1 <= radius:
                        queue.append((succ_id, distance + 1))

            if distance < radius:
                for pred_id in nodes[node_id].in_edges:
                    if pred_id in nodes and pred_id not in visited:
                        subgraph_edges.append((pred_id, node_id))
                        if distance + 1 <= radius:
                            queue.append((pred_id, distance + 1))

        return {
            'center': center_id,
            'nodes': list(subgraph_nodes.keys()),
            'edges': subgraph_edges,
            'radius': radius
        }

    def _subgraph_match_score(self, sg1: Dict[str, Any],
                              sg2: Dict[str, Any]) -> float:
        """Calculate match score between two subgraphs"""
        # Compare size
        size_sim = 1 - abs(len(sg1['nodes']) - len(sg2['nodes'])) / \
                   max(len(sg1['nodes']), len(sg2['nodes']), 1)

        edge_sim = 1 - abs(len(sg1['edges']) - len(sg2['edges'])) / \
                   max(len(sg1['edges']), len(sg2['edges']), 1)

        center_sim = 0.5

        return (size_sim * 0.4 + edge_sim * 0.4 + center_sim * 0.2)

    def _path_similarity(self, path1: List[int], path2: List[int]) -> float:
        """Calculate similarity between two execution paths"""
        if not path1 or not path2:
            return 0.0

        lcs_length = self._longest_common_subsequence_length(path1, path2)
        max_len = max(len(path1), len(path2))

        return lcs_length / max_len if max_len > 0 else 0.0

    def _longest_common_subsequence_length(self, seq1: List, seq2: List) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _get_node_type_distribution(self, cfg: ControlFlowGraph) -> Dict[str, int]:
        """Get distribution of node types"""
        type_dist = {}
        for node in cfg.nodes.values():
            node_type = node.type.value
            type_dist[node_type] = type_dist.get(node_type, 0) + 1

        return type_dist

    def _get_nodes_by_type(self, cfg: ControlFlowGraph,
                           node_type: NodeType) -> List[CFGNode]:
        """Get all nodes of specific type"""
        return [node for node in cfg.nodes.values() if node.type == node_type]

    def _decision_structure_similarity(self, cfg1: ControlFlowGraph,
                                       cfg2: ControlFlowGraph,
                                       node1: CFGNode, node2: CFGNode) -> float:
        """Calculate similarity between two decision structures"""
        branches1 = self._extract_branches(cfg1, node1.id)
        branches2 = self._extract_branches(cfg2, node2.id)

        if not branches1 and not branches2:
            return 1.0

        if len(branches1) != len(branches2):
            return 0.3

        similarity = 0
        for b1, b2 in zip(branches1, branches2):
            similarity += 1 - abs(len(b1) - len(b2)) / max(len(b1), len(b2), 1)

        return similarity / len(branches1) if branches1 else 0.0

    def _extract_branches(self, cfg: ControlFlowGraph,
                          decision_node_id: int) -> List[List[int]]:
        """Extract branches from a decision node"""
        branches = []
        node = cfg.nodes.get(decision_node_id)

        if not node or node.type != NodeType.DECISION:
            return branches

        for succ_id in node.out_edges:
            branch = self._extract_branch_path(cfg, succ_id, max_depth=5)
            if branch:
                branches.append(branch)

        return branches

    def _extract_branch_path(self, cfg: ControlFlowGraph,
                             start_id: int, max_depth: int = 5) -> List[int]:
        """Extract a single branch path"""
        path = []
        current_id = start_id
        depth = 0

        while current_id and depth < max_depth:
            path.append(current_id)
            node = cfg.nodes.get(current_id)

            if not node or not node.out_edges:
                break

            current_id = node.out_edges[0] if node.out_edges else None
            depth += 1

        return path

    def _loop_structure_similarity(self, cfg1: ControlFlowGraph,
                                   cfg2: ControlFlowGraph,
                                   node1: CFGNode, node2: CFGNode) -> float:
        """Calculate similarity between two loop structures"""
        # Extract loop body
        body1 = self._extract_loop_body(cfg1, node1.id)
        body2 = self._extract_loop_body(cfg2, node2.id)

        if not body1 and not body2:
            return 1.0

        size_sim = 1 - abs(len(body1) - len(body2)) / max(len(body1), len(body2), 1)

        nested1 = any(cfg1.nodes[nid].type in [NodeType.DECISION, NodeType.LOOP]
                      for nid in body1)
        nested2 = any(cfg2.nodes[nid].type in [NodeType.DECISION, NodeType.LOOP]
                      for nid in body2)

        structure_sim = 1.0 if nested1 == nested2 else 0.5

        return (size_sim * 0.7 + structure_sim * 0.3)

    def _extract_loop_body(self, cfg: ControlFlowGraph,
                           loop_node_id: int) -> List[int]:
        """Extract loop body nodes"""
        body = []
        visited = set()

        def dfs(node_id: int, depth: int = 0):
            if depth > 10 or node_id in visited:
                return

            visited.add(node_id)
            body.append(node_id)

            node = cfg.nodes.get(node_id)
            if node:
                for succ_id in node.out_edges:
                    if succ_id != loop_node_id:
                        dfs(succ_id, depth + 1)

        loop_node = cfg.nodes.get(loop_node_id)
        if loop_node:
            for succ_id in loop_node.out_edges:
                dfs(succ_id)

        return body

    def _function_structure_similarity(self, cfg1: ControlFlowGraph,
                                       cfg2: ControlFlowGraph,
                                       node1: CFGNode, node2: CFGNode) -> float:
        """Calculate similarity between two function structures"""
        body1 = self._extract_function_body(cfg1, node1.id)
        body2 = self._extract_function_body(cfg2, node2.id)

        size_sim = 1 - abs(len(body1) - len(body2)) / max(len(body1), len(body2), 1)

        control_count1 = self._count_control_structures(cfg1, body1)
        control_count2 = self._count_control_structures(cfg2, body2)

        control_sim = 1 - abs(control_count1 - control_count2) / \
                      max(control_count1, control_count2, 1)

        return (size_sim * 0.6 + control_sim * 0.4)

    def _extract_function_body(self, cfg: ControlFlowGraph,
                               func_node_id: int) -> List[int]:
        """Extract function body nodes"""
        body = []
        visited = set()

        def dfs(node_id: int):
            if node_id in visited:
                return

            visited.add(node_id)
            body.append(node_id)

            node = cfg.nodes.get(node_id)
            if node and node.type != NodeType.RETURN:
                for succ_id in node.out_edges:
                    dfs(succ_id)

        func_node = cfg.nodes.get(func_node_id)
        if func_node:
            for succ_id in func_node.out_edges:
                dfs(succ_id)

        return body

    def _count_control_structures(self, cfg: ControlFlowGraph,
                                  node_ids: List[int]) -> int:
        """Count control structures in a set of nodes"""
        count = 0
        for node_id in node_ids:
            node = cfg.nodes.get(node_id)
            if node and node.type in [NodeType.DECISION, NodeType.LOOP]:
                count += 1

        return count

    def _compare_dictionaries(self, dict1: Dict, dict2: Dict) -> float:
        """Compare two dictionaries"""
        all_keys = set(list(dict1.keys()) + list(dict2.keys()))
        if not all_keys:
            return 1.0

        similarity = 0
        for key in all_keys:
            val1 = dict1.get(key, 0)
            val2 = dict2.get(key, 0)

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if val1 == 0 and val2 == 0:
                    sim = 1.0
                else:
                    sim = 1 - abs(val1 - val2) / max(val1, val2)
                similarity += sim

        return similarity / len(all_keys)