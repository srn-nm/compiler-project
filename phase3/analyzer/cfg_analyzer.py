"""
Control Flow Graph (CFG) Analyzer - Phase 3
Complete implementation with Phase 1 & 2 integration
"""

import json
import numpy as np
from collections import deque
from typing import Dict, List, Set, Any, Tuple, Optional
from pathlib import Path

from .cfg_builder import ControlFlowGraph, CFGBuilder, CFGNode, NodeType
from phase3.analyzer.graph_similarity import GraphSimilarity
from ..utils.helpers import normalize_code, save_json, load_json


class CFGAnalyzer:
    """Main CFG analyzer class with full phase integration"""

    def __init__(self, language: str = 'python', config_path: str = None):
        self.language = language
        self.config = self._load_config(config_path)
        self.builder = CFGBuilder(language)
        self.graph_similarity = GraphSimilarity()

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration with defaults"""
        default_config = {
            'plagiarism_threshold': 0.7,
            'integration_weights': {
                'token': 0.2,
                'ast': 0.3,
                'cfg': 0.5
            },
            'similarity_weights': {
                'structural': 0.35,
                'graph_edit': 0.35,
                'subgraph': 0.30
            },
            'matching_threshold': 0.6,
            'max_paths': 50,
            'neighborhood_radius': 2
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # Deep update
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                print(f"⚠️ Error loading config: {e}")

        return default_config

    def _count_ast_nodes(self, ast_node: Dict) -> int:
        """Count nodes in AST dictionary"""
        if not ast_node:
            return 0
        count = 1
        for child in ast_node.get('children', []):
            count += self._count_ast_nodes(child)
        return count

    def build_cfg_from_ast(self, ast_data: Dict) -> ControlFlowGraph:
        """Build CFG from AST data with error handling"""
        try:
            cfg = ControlFlowGraph(self.language)
            return cfg.build_from_ast(ast_data)
        except Exception as e:
            print(f"⚠️ Error building CFG: {e}")
            # Return empty CFG
            return ControlFlowGraph(self.language)

    def _get_asts_from_phase2(self, code1: str, code2: str, 
                              phase2_results: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Get ASTs from Phase 2 results or create mock ASTs"""
        if phase2_results:
            # Try to get real AST from phase 2
            ast1 = phase2_results.get('ast1_dict')
            ast2 = phase2_results.get('ast2_dict')
            
            if ast1 and ast2:
                print(f"   📦 AST1: {self._count_ast_nodes(ast1)} nodes")
                print(f"   📦 AST2: {self._count_ast_nodes(ast2)} nodes")
                return ast1, ast2
        
        # Create mock ASTs for testing
        from .cfg_builder import create_mock_ast
        print("   ⚠️ Using mock AST (Phase 2 AST not available)")
        return create_mock_ast(), create_mock_ast()

    def analyze_code_pair(self, code1: str, code2: str,
                          ast1: Optional[Dict] = None,
                          ast2: Optional[Dict] = None,
                          phase2_results: Optional[Dict] = None) -> Dict[str, Any]:
        """Complete CFG analysis with all metrics"""
        
        # Get ASTs
        if ast1 is None or ast2 is None:
            ast1, ast2 = self._get_asts_from_phase2(code1, code2, phase2_results)

        # Build CFGs
        print(f"   🏗️  Building CFG1...")
        cfg1 = self.build_cfg_from_ast(ast1)
        print(f"   🏗️  Building CFG2...")
        cfg2 = self.build_cfg_from_ast(ast2)
        
        print(f"   🔗 CFG1: {len(cfg1.nodes)} nodes, {len(cfg1.edges)} edges")
        print(f"   🔗 CFG2: {len(cfg2.nodes)} nodes, {len(cfg2.edges)} edges")

        # Calculate all similarity metrics
        metrics = {}
        
        # 1. Structural similarity
        metrics['structural_similarity'] = self.structural_similarity(cfg1, cfg2)
        
        # 2. Graph edit distance
        metrics['graph_edit_similarity'] = self.graph_edit_distance(cfg1, cfg2)
        
        # 3. Subgraph matching
        metrics['subgraph_similarity'] = self.subgraph_matching(cfg1, cfg2)
        
        # 4. Path similarity
        metrics['path_similarity'] = self.path_similarity(cfg1, cfg2)
        
        # 5. Node type distribution
        metrics['node_type_similarity'] = self.node_type_similarity(cfg1, cfg2)

        # Calculate overall score
        weights = self.config['similarity_weights']
        overall_score = (
            weights['structural'] * metrics['structural_similarity'] +
            weights['graph_edit'] * metrics['graph_edit_similarity'] +
            weights['subgraph'] * metrics['subgraph_similarity']
        ) * 100

        # Extract statistics
        cfg_stats1 = self._extract_cfg_statistics(cfg1)
        cfg_stats2 = self._extract_cfg_statistics(cfg2)
        
        # Find matching components
        similar_components = self.graph_similarity.find_similar_components(cfg1, cfg2)
        
        # Find matching paths
        matched_paths = self._find_matching_paths(cfg1, cfg2)

        return {
            'cfg_similarity_score': round(overall_score, 2),
            'cfg_similarity_metrics': {k: round(v, 4) for k, v in metrics.items()},
            'cfg_statistics': {
                'code1': cfg_stats1,
                'code2': cfg_stats2
            },
            'similar_components': similar_components[:10],
            'matched_paths_count': len(matched_paths),
            'matched_paths_sample': matched_paths[:5],
            'is_plagiarism_suspected': overall_score >= (self.config['plagiarism_threshold'] * 100),
            'threshold_used': self.config['plagiarism_threshold'],
            'config_used': {
                'weights': self.config['similarity_weights'],
                'threshold': self.config['plagiarism_threshold']
            }
        }

    def structural_similarity(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph) -> float:
        """Structural similarity based on graph features"""
        features1 = self._extract_structural_features(cfg1)
        features2 = self._extract_structural_features(cfg2)

        similarity = 0.0
        total_features = 0

        # Compare numerical features
        numerical_features = ['node_count', 'edge_count', 'cyclomatic_complexity',
                             'avg_in_degree', 'avg_out_degree']
        
        for feature in numerical_features:
            if feature in features1 and feature in features2:
                val1 = features1[feature]
                val2 = features2[feature]
                if val1 == 0 and val2 == 0:
                    sim = 1.0
                else:
                    sim = 1 - abs(val1 - val2) / max(val1, val2, 1)
                similarity += sim
                total_features += 1

        # Compare type distributions
        if 'type_distribution' in features1 and 'type_distribution' in features2:
            sim = self._compare_distributions(
                features1['type_distribution'],
                features2['type_distribution']
            )
            similarity += sim
            total_features += 1

        return similarity / total_features if total_features > 0 else 0.0

    def graph_edit_distance(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph) -> float:
        """Graph Edit Distance similarity"""
        n1 = len(cfg1.nodes)
        n2 = len(cfg2.nodes)

        if n1 == 0 and n2 == 0:
            return 1.0

        # Create similarity matrix
        nodes1 = list(cfg1.nodes.values())
        nodes2 = list(cfg2.nodes.values())
        sim_matrix = np.zeros((n1, n2))

        for i, node1 in enumerate(nodes1):
            for j, node2 in enumerate(nodes2):
                sim_matrix[i, j] = self._node_similarity(node1, node2)

        # Find optimal matching
        matched_pairs = self._find_optimal_matching(sim_matrix)

        # Calculate edit cost
        edit_cost = 0.0
        matched_nodes1 = set(i for i, _ in matched_pairs)
        matched_nodes2 = set(j for _, j in matched_pairs)

        # Node insertion/deletion cost
        edit_cost += (n1 - len(matched_nodes1)) * 0.5
        edit_cost += (n2 - len(matched_nodes2)) * 0.5

        # Node substitution cost
        for i, j in matched_pairs:
            edit_cost += 1 - sim_matrix[i, j]

        # Edge edit cost
        edge_cost = self._calculate_edge_cost(cfg1, cfg2, matched_pairs, nodes1, nodes2)
        edit_cost += edge_cost * 0.3

        # Normalize similarity
        max_cost = (n1 + n2) * 0.5 + (len(cfg1.edges) + len(cfg2.edges)) * 0.3
        similarity = 1 - (edit_cost / max_cost) if max_cost > 0 else 1.0

        return max(0.0, min(1.0, similarity))

    def subgraph_matching(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph) -> float:
        """Subgraph matching similarity for control structures"""
        # Extract important subgraphs (decisions, loops, functions)
        subgraphs1 = self._extract_control_subgraphs(cfg1)
        subgraphs2 = self._extract_control_subgraphs(cfg2)

        if not subgraphs1 and not subgraphs2:
            return 1.0

        # Match subgraphs
        matched = 0
        used_indices = set()

        for sg1 in subgraphs1:
            best_match = 0.0
            best_idx = -1
            
            for i, sg2 in enumerate(subgraphs2):
                if i in used_indices:
                    continue
                    
                score = self._subgraph_similarity(sg1, sg2)
                if score > best_match:
                    best_match = score
                    best_idx = i
            
            if best_match >= 0.7:
                matched += 1
                if best_idx != -1:
                    used_indices.add(best_idx)

        total = max(len(subgraphs1), len(subgraphs2))
        return matched / total if total > 0 else 0.0

    def path_similarity(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph) -> float:
        """Execution path similarity"""
        paths1 = cfg1.get_execution_paths(max_paths=self.config['max_paths'])
        paths2 = cfg2.get_execution_paths(max_paths=self.config['max_paths'])

        if not paths1 and not paths2:
            return 1.0
        if not paths1 or not paths2:
            return 0.0

        # Compare path signatures
        signatures1 = [self._get_path_signature(p, cfg1) for p in paths1[:10]]
        signatures2 = [self._get_path_signature(p, cfg2) for p in paths2[:10]]

        similarity = 0.0
        comparisons = 0

        for sig1 in signatures1:
            best = 0.0
            for sig2 in signatures2:
                sim = self._signature_similarity(sig1, sig2)
                best = max(best, sim)
            similarity += best
            comparisons += 1

        return similarity / comparisons if comparisons > 0 else 0.0

    def node_type_similarity(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph) -> float:
        """Node type distribution similarity"""
        dist1 = self._get_node_type_distribution(cfg1)
        dist2 = self._get_node_type_distribution(cfg2)
        return self._compare_distributions(dist1, dist2)

    # ============ Helper Methods ============

    def _extract_structural_features(self, cfg: ControlFlowGraph) -> Dict[str, Any]:
        """Extract comprehensive structural features"""
        nodes = cfg.nodes
        if not nodes:
            return {
                'node_count': 0, 'edge_count': 0,
                'avg_in_degree': 0, 'avg_out_degree': 0,
                'type_distribution': {}, 'cyclomatic_complexity': 0
            }

        in_degrees = [len(node.in_edges) for node in nodes.values()]
        out_degrees = [len(node.out_edges) for node in nodes.values()]

        return {
            'node_count': len(nodes),
            'edge_count': len(cfg.edges),
            'avg_in_degree': float(np.mean(in_degrees)) if in_degrees else 0,
            'avg_out_degree': float(np.mean(out_degrees)) if out_degrees else 0,
            'type_distribution': self._get_node_type_distribution(cfg),
            'cyclomatic_complexity': cfg.get_cyclomatic_complexity()
        }

    def _extract_cfg_statistics(self, cfg: ControlFlowGraph) -> Dict[str, Any]:
        """Extract detailed CFG statistics"""
        stats = {
            'node_count': len(cfg.nodes),
            'edge_count': len(cfg.edges),
            'entry_node': cfg.entry_node_id,
            'exit_node': cfg.exit_node_id,
            'cyclomatic_complexity': cfg.get_cyclomatic_complexity(),
            'execution_paths': len(cfg.get_execution_paths(max_paths=100)),
            'control_structures': cfg.get_control_structures_count()
        }
        return stats

    def _node_similarity(self, node1: CFGNode, node2: CFGNode) -> float:
        """Calculate similarity between two CFG nodes"""
        similarity = 0.0
        
        # Type match (40%)
        if node1.type == node2.type:
            similarity += 0.4
        
        # Degree similarity (30%)
        in_degree_sim = 1 - abs(len(node1.in_edges) - len(node2.in_edges)) / \
                       max(len(node1.in_edges), len(node2.in_edges), 1)
        out_degree_sim = 1 - abs(len(node1.out_edges) - len(node2.out_edges)) / \
                        max(len(node1.out_edges), len(node2.out_edges), 1)
        
        similarity += 0.15 * in_degree_sim
        similarity += 0.15 * out_degree_sim
        
        # Label similarity (30%)
        if node1.label and node2.label:
            label1 = node1.label.lower()
            label2 = node2.label.lower()
            
            if label1 == label2:
                similarity += 0.3
            elif any(k in label1 and k in label2 for k in ['if', 'while', 'for', 'return']):
                similarity += 0.15
        
        return min(1.0, similarity)

    def _find_optimal_matching(self, sim_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Greedy optimal matching"""
        n1, n2 = sim_matrix.shape
        matched = []
        used_rows = set()
        used_cols = set()
        
        # Create list of all potential matches
        candidates = []
        for i in range(n1):
            for j in range(n2):
                candidates.append((sim_matrix[i, j], i, j))
        
        # Sort by similarity descending
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Greedy selection
        for sim, i, j in candidates:
            if sim >= self.config['matching_threshold']:
                if i not in used_rows and j not in used_cols:
                    matched.append((i, j))
                    used_rows.add(i)
                    used_cols.add(j)
        
        return matched

    def _calculate_edge_cost(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph,
                            matched_pairs: List[Tuple[int, int]],
                            nodes1: List[CFGNode], nodes2: List[CFGNode]) -> float:
        """Calculate edge edit cost"""
        if not matched_pairs:
            return 0.0

        # Create node ID mapping
        mapping = {}
        for i, j in matched_pairs:
            mapping[nodes1[i].id] = nodes2[j].id

        # Count mismatched edges
        mismatched = 0
        for from_id, to_id, _ in cfg1.edges:
            if from_id in mapping and to_id in mapping:
                mapped_from = mapping[from_id]
                mapped_to = mapping[to_id]
                
                # Check if edge exists in cfg2
                edge_exists = any(
                    f == mapped_from and t == mapped_to 
                    for f, t, _ in cfg2.edges
                )
                if not edge_exists:
                    mismatched += 1

        total_edges = max(len(cfg1.edges), len(cfg2.edges), 1)
        return mismatched / total_edges

    def _extract_control_subgraphs(self, cfg: ControlFlowGraph) -> List[Dict]:
        """Extract subgraphs for control structures"""
        subgraphs = []
        radius = self.config['neighborhood_radius']
        
        for node_id, node in cfg.nodes.items():
            if node.type in [NodeType.DECISION, NodeType.LOOP, NodeType.FUNCTION]:
                subgraph = self._extract_neighborhood(cfg, node_id, radius)
                if subgraph:
                    subgraphs.append(subgraph)
        
        return subgraphs

    def _extract_neighborhood(self, cfg: ControlFlowGraph, 
                             center_id: int, radius: int) -> Optional[Dict]:
        """Extract neighborhood around a node"""
        if center_id not in cfg.nodes:
            return None

        nodes = cfg.nodes
        visited = set()
        queue = deque([(center_id, 0)])
        subgraph_nodes = []
        subgraph_edges = []

        while queue:
            node_id, distance = queue.popleft()
            
            if node_id in visited or distance > radius:
                continue
                
            visited.add(node_id)
            subgraph_nodes.append(node_id)
            
            # Add outgoing edges
            for succ_id in nodes[node_id].out_edges:
                if succ_id in nodes:
                    subgraph_edges.append((node_id, succ_id))
                    if succ_id not in visited and distance + 1 <= radius:
                        queue.append((succ_id, distance + 1))
            
            # Add incoming edges (undirected for neighborhood)
            if distance < radius:
                for pred_id in nodes[node_id].in_edges:
                    if pred_id in nodes and pred_id not in visited:
                        subgraph_edges.append((pred_id, node_id))
                        queue.append((pred_id, distance + 1))

        return {
            'center': center_id,
            'center_type': nodes[center_id].type.value,
            'nodes': subgraph_nodes,
            'edges': subgraph_edges,
            'radius': radius
        }

    def _subgraph_similarity(self, sg1: Dict, sg2: Dict) -> float:
        """Calculate similarity between two subgraphs"""
        # Size similarity
        size_sim = 1 - abs(len(sg1['nodes']) - len(sg2['nodes'])) / \
                  max(len(sg1['nodes']), len(sg2['nodes']), 1)
        
        # Edge similarity
        edge_sim = 1 - abs(len(sg1['edges']) - len(sg2['edges'])) / \
                  max(len(sg1['edges']), len(sg2['edges']), 1)
        
        # Type similarity
        type_sim = 1.0 if sg1['center_type'] == sg2['center_type'] else 0.5
        
        return size_sim * 0.4 + edge_sim * 0.3 + type_sim * 0.3

    def _find_matching_paths(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph) -> List[Dict]:
        """Find matching execution paths"""
        paths1 = cfg1.get_execution_paths(max_paths=self.config['max_paths'])
        paths2 = cfg2.get_execution_paths(max_paths=self.config['max_paths'])
        
        matches = []
        for i, p1 in enumerate(paths1[:10]):
            for j, p2 in enumerate(paths2[:10]):
                # Length similarity
                len_sim = 1 - abs(len(p1) - len(p2)) / max(len(p1), len(p2), 1)
                if len_sim > 0.8:
                    matches.append({
                        'path1_index': i,
                        'path2_index': j,
                        'length1': len(p1),
                        'length2': len(p2),
                        'similarity': round(len_sim, 3)
                    })
        
        return matches[:10]

    def _get_node_type_distribution(self, cfg: ControlFlowGraph) -> Dict[str, int]:
        """Get distribution of node types"""
        dist = {}
        for node in cfg.nodes.values():
            t = node.type.value
            dist[t] = dist.get(t, 0) + 1
        return dist

    def _get_path_signature(self, path: List[int], cfg: ControlFlowGraph) -> List[str]:
        """Create signature for execution path"""
        signature = []
        for node_id in path[:10]:
            if node_id in cfg.nodes:
                node = cfg.nodes[node_id]
                signature.append(f"{node.type.value}")
        return signature

    def _signature_similarity(self, sig1: List[str], sig2: List[str]) -> float:
        """Compare two path signatures"""
        if not sig1 or not sig2:
            return 0.0
        
        # Longest common subsequence
        m, n = len(sig1), len(sig2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if sig1[i-1] == sig2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs = dp[m][n]
        return lcs / max(m, n)

    def _compare_distributions(self, dist1: Dict, dist2: Dict) -> float:
        """Compare two distribution dictionaries"""
        all_keys = set(dist1.keys()) | set(dist2.keys())
        if not all_keys:
            return 1.0
        
        vec1 = [dist1.get(k, 0) for k in all_keys]
        vec2 = [dist2.get(k, 0) for k in all_keys]
        
        # Cosine similarity
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = np.sqrt(sum(a * a for a in vec1))
        norm2 = np.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)


class Phase3CFGSimilarity:
    """Wrapper class for CFG analysis with phase integration"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.analyzer = CFGAnalyzer(config_path=config_path)

    def analyze_code_pair(self, code1: str, code2: str,
                          phase1_results: Optional[Dict] = None,
                          phase2_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete analysis with Phase 1 & 2 integration
        """
        # Get ASTs from Phase 2
        ast1 = None
        ast2 = None
        
        if phase2_results:
            ast1 = phase2_results.get('ast1_dict')
            ast2 = phase2_results.get('ast2_dict')

        # Run CFG analysis
        cfg_results = self.analyzer.analyze_code_pair(
            code1, code2, ast1, ast2, phase2_results
        )

        # Integrate all phases if both results available
        if phase1_results and phase2_results:
            from ..integration.phase_integration import Phase3Integration
            return Phase3Integration.integrate_all_phases(
                code1, code2, phase1_results, phase2_results, cfg_results
            )

        return cfg_results

    def analyze_files(self, file1: str, file2: str,
                      language: str = 'python') -> Dict[str, Any]:
        """Analyze two code files directly"""
        with open(file1, 'r', encoding='utf-8') as f:
            code1 = f.read()
        with open(file2, 'r', encoding='utf-8') as f:
            code2 = f.read()
        
        return self.analyze_code_pair(code1, code2)