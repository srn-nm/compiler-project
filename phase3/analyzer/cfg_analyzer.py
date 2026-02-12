"""
Control Flow Graph (CFG) Analyzer - Phase 3
"""

import json
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Set, Any, Tuple, Optional
from pathlib import Path

from .cfg_builder import ControlFlowGraph, CFGBuilder, CFGNode, BasicBlock, NodeType

from .graph_similarity import GraphSimilarity
from ..utils.utils import normalize_code, calculate_hash, format_percentage

class CFGAnalyzer:
    """Main CFG analyzer class"""

    def __init__(self, language: str = 'python', config_path: str = None):
        self.language = language
        self.config = self._load_config(config_path)
        self.builder = CFGBuilder(language)
        self.graph_similarity = GraphSimilarity()
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration"""
        default_config = {
            'plagiarism_threshold': 0.7,
            'integration_weights': {
                'token': 0.2,
                'ast': 0.3,
                'cfg': 0.5
            },
            'similarity_weights': {
                'structural': 0.4,
                'graph_edit': 0.3,
                'subgraph': 0.3
            }
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}")

        return default_config

    def build_cfg_from_ast(self, ast_data: Dict) -> ControlFlowGraph:
        """Build CFG from AST data"""
        return self.builder.build_from_ast(ast_data)

    def _get_asts_from_phase2(self, code1: str, code2: str) -> Tuple[Dict, Dict]:
        """Get ASTs from phase 2 or create mock ASTs"""
        return {
            'type': 'Module',
            'children': []
        }, {
            'type': 'Module',
            'children': []
        }

    def analyze_code_pair(self, code1: str, code2: str,
                          ast1: Optional[Dict] = None,
                          ast2: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze similarity between two codes using CFG"""

        if ast1 is None or ast2 is None:
            ast1, ast2 = self._get_asts_from_phase2(code1, code2)

        cfg1 = self.build_cfg_from_ast(ast1)
        cfg2 = self.build_cfg_from_ast(ast2)

        metrics = {}

        structural_sim = self.structural_similarity(cfg1, cfg2)
        metrics['structural_similarity'] = structural_sim

        graph_edit_sim = self.graph_edit_distance(cfg1, cfg2)
        metrics['graph_edit_similarity'] = graph_edit_sim

        subgraph_sim = self.subgraph_matching(cfg1, cfg2)
        metrics['subgraph_similarity'] = subgraph_sim

        weights = self.config['similarity_weights']
        overall_score = (
                                weights['structural'] * structural_sim +
                                weights['graph_edit'] * graph_edit_sim +
                                weights['subgraph'] * subgraph_sim
                        ) * 100

        cfg_stats1 = self._extract_cfg_statistics(cfg1)
        cfg_stats2 = self._extract_cfg_statistics(cfg2)

        matched_paths = self._find_matching_paths(cfg1, cfg2)

        return {
            'cfg_similarity_score': overall_score,
            'cfg_similarity_metrics': metrics,
            'cfg_statistics': {
                'code1': cfg_stats1,
                'code2': cfg_stats2
            },
            'matched_paths_count': len(matched_paths),
            'matched_paths_sample': matched_paths[:5],
            'is_plagiarism_suspected': overall_score >= (self.config['plagiarism_threshold'] * 100),
            'threshold_used': self.config['plagiarism_threshold']
        }

    def structural_similarity(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph) -> float:
        """Structural similarity of two graphs"""
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
                    sim = self._compare_feature_dicts(val1, val2)
                    similarity += sim
                    total_features += 1

        return similarity / total_features if total_features > 0 else 0.0

    def graph_edit_distance(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph) -> float:
        """Graph Edit Distance similarity"""
        n1 = len(cfg1.nodes)
        n2 = len(cfg2.nodes)

        node_similarity = np.zeros((n1, n2))
        nodes1 = list(cfg1.nodes.values())
        nodes2 = list(cfg2.nodes.values())

        for i, node1 in enumerate(nodes1):
            for j, node2 in enumerate(nodes2):
                node_similarity[i, j] = self._node_similarity(node1, node2)

        matched_pairs = self._find_best_matching(node_similarity)

        edit_cost = 0
        total_nodes = max(n1, n2)

        matched_nodes1 = set(i for i, _ in matched_pairs)
        matched_nodes2 = set(j for _, j in matched_pairs)

        edit_cost += (n1 - len(matched_nodes1))
        edit_cost += (n2 - len(matched_nodes2))

        for i, j in matched_pairs:
            edit_cost += 1 - node_similarity[i, j]

        max_cost = total_nodes * 2
        similarity = 1 - (edit_cost / max_cost) if max_cost > 0 else 1.0

        return max(similarity, 0.0)

    def subgraph_matching(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph) -> float:
        """Subgraph matching similarity"""
        subgraphs1 = self._extract_important_subgraphs(cfg1)
        subgraphs2 = self._extract_important_subgraphs(cfg2)

        if not subgraphs1 and not subgraphs2:
            return 1.0

        matched_subgraphs = 0
        for sg1 in subgraphs1:
            for sg2 in subgraphs2:
                if self._subgraphs_similar(sg1, sg2):
                    matched_subgraphs += 1
                    break

        total_subgraphs = max(len(subgraphs1), len(subgraphs2))
        return matched_subgraphs / total_subgraphs if total_subgraphs > 0 else 0.0

    def _extract_structural_features(self, cfg: ControlFlowGraph) -> Dict[str, Any]:
        """Extract graph structural features"""
        nodes = cfg.nodes

        in_degrees = [len(node.in_edges) for node in nodes.values()]
        out_degrees = [len(node.out_edges) for node in nodes.values()]

        type_dist = {}
        for node in nodes.values():
            node_type = node.type.value
            type_dist[node_type] = type_dist.get(node_type, 0) + 1

        return {
            'node_count': len(nodes),
            'edge_count': len(cfg.edges),
            'avg_in_degree': np.mean(in_degrees) if in_degrees else 0,
            'avg_out_degree': np.mean(out_degrees) if out_degrees else 0,
            'type_distribution': type_dist,
            'cyclomatic_complexity': cfg.get_cyclomatic_complexity()
        }

    def _node_similarity(self, node1: CFGNode, node2: CFGNode) -> float:
        """Similarity between two nodes"""
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
            if node1.label == node2.label:
                similarity += 0.2
            elif node1.label.lower() == node2.label.lower():
                similarity += 0.1

        return min(similarity, 1.0)

    def _find_best_matching(self, similarity_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Find best matching between nodes"""
        n1, n2 = similarity_matrix.shape
        matched_pairs = []

        threshold = 0.6

        for i in range(n1):
            best_j = -1
            best_sim = threshold

            for j in range(n2):
                if similarity_matrix[i, j] > best_sim:
                    if not any(j == pair[1] for pair in matched_pairs):
                        best_j = j
                        best_sim = similarity_matrix[i, j]

            if best_j != -1:
                matched_pairs.append((i, best_j))

        return matched_pairs

    def _extract_important_subgraphs(self, cfg: ControlFlowGraph) -> List[Dict]:
        """Extract important subgraphs"""
        subgraphs = []
        nodes = cfg.nodes

        for node_id, node in nodes.items():
            if node.type.value in ['DECISION', 'LOOP']:
                subgraph = self._extract_neighborhood(cfg, node_id, radius=2)
                if subgraph:
                    subgraphs.append(subgraph)

        return subgraphs

    def _extract_neighborhood(self, cfg: ControlFlowGraph, center_id: int, radius: int = 2) -> Dict:
        """Extract neighborhood of a node"""
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

        return {
            'center': center_id,
            'nodes_count': len(subgraph_nodes),
            'edges_count': len(subgraph_edges)
        }

    def _subgraphs_similar(self, sg1: Dict, sg2: Dict) -> bool:
        """Check similarity of two subgraphs"""
        nodes1 = sg1.get('nodes_count', 0)
        nodes2 = sg2.get('nodes_count', 0)
        edges1 = sg1.get('edges_count', 0)
        edges2 = sg2.get('edges_count', 0)

        if abs(nodes1 - nodes2) > max(nodes1, nodes2) * 0.5:
            return False

        if abs(edges1 - edges2) > max(edges1, edges2) * 0.5:
            return False

        return True

    def _compare_feature_dicts(self, dict1: Dict, dict2: Dict) -> float:
        """Compare feature dictionaries"""
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

    def _extract_cfg_statistics(self, cfg: ControlFlowGraph) -> Dict[str, Any]:
        """Extract statistics from CFG"""
        stats = cfg.to_dict()

        stats['execution_paths'] = len(cfg.get_execution_paths(max_paths=100))
        stats['control_structures'] = cfg._count_control_structures()

        return stats

    def _find_matching_paths(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph) -> List[Dict]:
        """Find matching execution paths"""
        paths1 = cfg1.get_execution_paths(max_paths=50)
        paths2 = cfg2.get_execution_paths(max_paths=50)

        matched_paths = []

        for i, path1 in enumerate(paths1):
            for j, path2 in enumerate(paths2):
                if abs(len(path1) - len(path2)) <= 2:
                    similarity = min(len(path1), len(path2)) / max(len(path1), len(path2))

                    if similarity > 0.8:
                        matched_paths.append({
                            'path1_index': i,
                            'path2_index': j,
                            'length1': len(path1),
                            'length2': len(path2),
                            'similarity': similarity
                        })

        return matched_paths


class Phase3CFGSimilarity:
    """Wrapper class for CFG analysis with phase integration"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.analyzer = CFGAnalyzer(config_path=config_path)

    def analyze_code_pair(self, code1: str, code2: str,
                          phase1_results: Dict = None,
                          phase2_results: Dict = None) -> Dict[str, Any]:
        """
        Analyze two codes with optional phase 1 and 2 results

        Args:
            code1: First code
            code2: Second code
            phase1_results: Results from phase 1 (token analysis)
            phase2_results: Results from phase 2 (AST analysis)

        Returns:
            Analysis results dictionary
        """
        ast1 = None
        ast2 = None

        if phase2_results:
            ast1 = phase2_results.get('ast1')
            ast2 = phase2_results.get('ast2')

            if not ast1 and 'ast_statistics' in phase2_results:
                ast1 = self._create_mock_ast_from_phase2(phase2_results, 'code1')
                ast2 = self._create_mock_ast_from_phase2(phase2_results, 'code2')

        cfg_results = self.analyzer.analyze_code_pair(code1, code2, ast1, ast2)

        if phase1_results and phase2_results:
            return self._integrate_all_results(phase1_results, phase2_results, cfg_results)

        return cfg_results

    def _create_mock_ast_from_phase2(self, phase2_results: Dict,
                                     code_key: str = 'code1') -> Dict:
        """
        Create a mock AST structure from phase 2 statistics

        """
        # This is a simplified mock AST - in real implementation,
        # you should pass actual AST from phase 2
        stats = phase2_results.get('ast_statistics', {}).get(code_key, {})

        return {
            'type': 'Module',
            'children': [
                {
                    'type': 'MockAST',
                    'value': f'Generated from phase2 stats: {stats.get("total_nodes", 0)} nodes',
                    'line': 1,
                    'col': 1,
                    'children': []
                }
            ]
        }

    def _integrate_all_results(self, phase1_results: Dict,
                               phase2_results: Dict,
                               phase3_results: Dict) -> Dict[str, Any]:
        """
        Combine results from all three phases

        """
        token_score = self._extract_normalized_score(phase1_results, 'overall_similarity')
        ast_score = self._extract_normalized_score(phase2_results, 'ast_similarity_score')
        cfg_score = self._extract_normalized_score(phase3_results, 'cfg_similarity_score')

        weights = self.analyzer.config.get('integration_weights', {
            'token': 0.2,
            'ast': 0.3,
            'cfg': 0.5
        })

        combined_score = (
                                 weights['token'] * token_score +
                                 weights['ast'] * ast_score +
                                 weights['cfg'] * cfg_score
                         ) * 100

        is_plagiarism = combined_score >= (self.analyzer.config['plagiarism_threshold'] * 100)

        return {
            'phase_integration': 'all_phases_combined',
            'combined_similarity_score': combined_score,
            'individual_scores': {
                'token': token_score * 100,
                'ast': ast_score * 100,
                'cfg': cfg_score * 100
            },
            'weights_applied': weights,
            'final_decision': 'PLAGIARISM_SUSPECTED' if is_plagiarism else 'CLEAN',
            'is_plagiarism_suspected': is_plagiarism,
            'threshold': self.analyzer.config['plagiarism_threshold'],
            'phase1_summary': self._extract_phase_summary(phase1_results, 'token'),
            'phase2_summary': self._extract_phase_summary(phase2_results, 'ast'),
            'phase3_summary': self._extract_phase_summary(phase3_results, 'cfg'),
            'confidence': self._calculate_confidence(token_score, ast_score, cfg_score)
        }

    def _extract_normalized_score(self, results: Dict, key: str) -> float:
        """Extract and normalize score to 0-1 range"""
        score = results.get(key, 0)
        if isinstance(score, (int, float)):
            return score / 100 if score > 1 else score
        return 0.0

    def _extract_phase_summary(self, results: Dict, phase_type: str) -> Dict:
        """Create a summary for each phase"""
        if phase_type == 'token':
            return {
                'score': results.get('overall_similarity', 0),
                'token_count': results.get('token_counts', {}).get('common', 0),
                'matched_sections': len(results.get('matched_sections', []))
            }
        elif phase_type == 'ast':
            return {
                'score': results.get('ast_similarity_score', 0),
                'node_count': results.get('ast_statistics', {}).get('code1', {}).get('total_nodes', 0),
                'matched_nodes': results.get('matched_nodes_count', 0)
            }
        elif phase_type == 'cfg':
            return {
                'score': results.get('cfg_similarity_score', 0),
                'node_count': results.get('cfg_statistics', {}).get('code1', {}).get('node_count', 0),
                'matched_paths': results.get('matched_paths_count', 0)
            }
        else:
            return {}

    def _calculate_confidence(self, token_score: float,
                              ast_score: float,
                              cfg_score: float) -> float:
        """Calculate confidence in the analysis"""
        scores = [token_score, ast_score, cfg_score]

        mean = sum(scores) / 3
        variance = sum((s - mean) ** 2 for s in scores) / 3

        confidence = 1 - min(variance * 5, 1.0)
        confidence *= mean

        return confidence * 100

    def analyze_files(self, file1: str, file2: str,
                      language: str = 'python') -> Dict[str, Any]:
        """
        Analyze two files directly

        """
        # Read files
        with open(file1, 'r', encoding='utf-8') as f:
            code1 = f.read()

        with open(file2, 'r', encoding='utf-8') as f:
            code2 = f.read()

        return self.analyze_code_pair(code1, code2)

    def analyze_code_pair(self, code1: str, code2: str,
                          phase1_results: Dict = None,
                          phase2_results: Dict = None) -> Dict[str, Any]:
        """
        Analyze two codes with optional phase 1 and 2 results
        """
        ast1 = None
        ast2 = None

        if phase2_results:
            # تلاش برای دریافت AST واقعی از نتایج فاز ۲
            ast1 = phase2_results.get('ast1_dict')
            ast2 = phase2_results.get('ast2_dict')

            # اگر AST وجود نداشت، از روش قبلی استفاده کن
            if not ast1:
                ast1 = self._create_mock_ast_from_phase2(phase2_results, 'code1')
            if not ast2:
                ast2 = self._create_mock_ast_from_phase2(phase2_results, 'code2')

        cfg_results = self.analyzer.analyze_code_pair(code1, code2, ast1, ast2)

        if phase1_results and phase2_results:
            return self._integrate_all_results(phase1_results, phase2_results, cfg_results)

        return cfg_results