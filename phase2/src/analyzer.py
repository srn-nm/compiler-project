import ast
import hashlib
import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


class ASTNode:

    def __init__(self, node_type: str, value: str = None, children: List['ASTNode'] = None,
                 line: int = None, col: int = None):
        self.node_type = node_type
        self.value = value
        self.children = children if children is not None else []
        self.line = line
        self.col = col
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        content = f"{self.node_type}:"
        for child in self.children:
            content += child.hash
        return hashlib.md5(content.encode()).hexdigest()

    def structural_hash(self) -> str:
        content = ""
        for child in self.children:
            content += child.structural_hash()
        return hashlib.md5(content.encode()).hexdigest() if content else "leaf"

    def to_dict(self) -> Dict:
        return {
            'type': self.node_type,
            'value': self.value,
            'line': self.line,
            'col': self.col,
            'children': [child.to_dict() for child in self.children],
            'hash': self.hash,
            'structural_hash': self.structural_hash()
        }


class ASTSimilarityAnalyzer:

    def __init__(self, language: str = 'python', config_path: str = None):
        self.language = language
        self.config = self.load_config(config_path)

    def load_config(self, config_path: Optional[str] = None) -> Dict:
        default_config = {
            'normalize_identifiers': True,
            'normalize_literals': True,
            'ignore_comments': True,
            'ignore_whitespace': True,
            'ast_similarity_weights': {
                'structural': 0.4,
                'node_types': 0.3,
                'subtree_matching': 0.3
            },
            'integration_weights': {
                'token': 0.3,
                'ast': 0.7
            },
            'plagiarism_threshold': 0.65
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                print(f"Error reading config file: {e}")

        return default_config

    def parse_code(self, code: str) -> ASTNode:
        if self.language == 'python':
            return self.parse_python_ast(code)
        else:
            raise ValueError(f"Language {self.language} not supported")

    def parse_python_ast(self, code: str) -> ASTNode:
        try:
            tree = ast.parse(code)
            return self.convert_ast_node(tree)
        except SyntaxError as e:
            print(f" Syntax error in code parsing: {e}")
            return ASTNode('Module', children=[])

    def convert_ast_node(self, node, parent_type: str = None) -> ASTNode:
        if node is None:
            return ASTNode('None')

        node_type = type(node).__name__

        value = None
        if isinstance(node, ast.Name):
            value = 'VAR' if self.config.get('normalize_identifiers') else node.id
        elif isinstance(node, ast.FunctionDef):
            value = 'FUNC' if self.config.get('normalize_identifiers') else node.name
        elif isinstance(node, ast.ClassDef):
            value = 'CLASS' if self.config.get('normalize_identifiers') else node.name
        elif isinstance(node, ast.Constant):
            if self.config.get('normalize_literals'):
                value = type(node.value).__name__.upper()  
            else:
                value = repr(node.value)
        elif hasattr(node, 'name'):
            value = node.name
        elif hasattr(node, 'id'):
            value = node.id

        # Extract position
        line = getattr(node, 'lineno', None)
        col = getattr(node, 'col_offset', None)

        ast_node = ASTNode(node_type, value, line=line, col=col)

        children = []

        if isinstance(node, ast.AST):
            for field_name in node._fields:
                field_value = getattr(node, field_name, None)

                if field_value is None:
                    continue

                if isinstance(field_value, list):
                    for item in field_value:
                        if isinstance(item, ast.AST):
                            child_node = self.convert_ast_node(item, node_type)
                            if child_node:
                                children.append(child_node)
                        elif field_value:
                            children.append(ASTNode(field_name, str(item)))
                elif isinstance(field_value, ast.AST):
                    child_node = self.convert_ast_node(field_value, node_type)
                    if child_node:
                        children.append(child_node)
                elif field_value:
                    children.append(ASTNode(field_name, str(field_value)))

        ast_node.children = children
        return ast_node

    def analyze_with_phase1(self, code1: str, code2: str, phase1_results: Dict = None) -> Dict[str, Any]:
        # Parameters: code1 code2
        # Build ASTs
        ast1 = self.parse_code(code1)
        ast2 = self.parse_code(code2)

        ast_similarity = self.calculate_ast_similarity(ast1, ast2)

        if phase1_results:
            return self.integrate_results(phase1_results, ast_similarity, code1, code2)
        else:
            return ast_similarity

    def calculate_ast_similarity(self, ast1: ASTNode, ast2: ASTNode) -> Dict[str, Any]:
        metrics = {}

        structural_sim = self.structural_similarity(ast1, ast2)
        metrics['structural_similarity'] = structural_sim

        node_type_sim = self.node_type_distribution_similarity(ast1, ast2)
        metrics['node_type_similarity'] = node_type_sim

        subtree_sim = self.subtree_matching_similarity(ast1, ast2)
        metrics['subtree_similarity'] = subtree_sim

        depth_sim = self.tree_depth_similarity(ast1, ast2)
        metrics['depth_similarity'] = depth_sim

        weights = self.config.get('ast_similarity_weights', {
            'structural': 0.4,
            'node_types': 0.3,
            'subtree_matching': 0.3
        })

        overall_score = (
                weights.get('structural', 0.4) * structural_sim +
                weights.get('node_types', 0.3) * node_type_sim +
                weights.get('subtree_matching', 0.3) * subtree_sim
        )

        ast1_stats = self.get_tree_statistics(ast1)
        ast2_stats = self.get_tree_statistics(ast2)

        matched_nodes = self.find_similar_nodes(ast1, ast2)

        return {
            'ast_similarity_score': overall_score * 100,
            'ast_similarity_metrics': metrics,
            'ast_statistics': {
                'code1': ast1_stats,
                'code2': ast2_stats
            },
            'matched_nodes_count': len(matched_nodes),
            'matched_nodes_sample': matched_nodes[:10],  # Only first 10 samples
            'is_plagiarism_suspected': overall_score >= self.config['plagiarism_threshold'],
            'threshold_used': self.config['plagiarism_threshold']
        }

    def structural_similarity(self, ast1: ASTNode, ast2: ASTNode) -> float:

        def collect_structural_hashes(node: ASTNode, hashes: List[str]):
            hashes.append(node.structural_hash())
            for child in node.children:
                collect_structural_hashes(child, hashes)

        hashes1 = []
        hashes2 = []
        collect_structural_hashes(ast1, hashes1)
        collect_structural_hashes(ast2, hashes2)

        set1 = set(hashes1)
        set2 = set(hashes2)

        if not set1 and not set2:
            return 1.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def node_type_distribution_similarity(self, ast1: ASTNode, ast2: ASTNode) -> float:
        from collections import Counter

        def count_node_types(node: ASTNode, counter: Counter):
            counter[node.node_type] += 1
            for child in node.children:
                count_node_types(child, counter)

        counter1 = Counter()
        counter2 = Counter()
        count_node_types(ast1, counter1)
        count_node_types(ast2, counter2)

        all_types = set(list(counter1.keys()) + list(counter2.keys()))

        if not all_types:
            return 1.0

        vec1 = [counter1.get(t, 0) for t in all_types]
        vec2 = [counter2.get(t, 0) for t in all_types]

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def subtree_matching_similarity(self, ast1: ASTNode, ast2: ASTNode) -> float:

        def collect_subtree_hashes(node: ASTNode, hashes: List[str], depth: int = 0, max_depth: int = 3):
            if depth > max_depth:
                return

            hashes.append(node.hash)
            for child in node.children:
                collect_subtree_hashes(child, hashes, depth + 1, max_depth)

        hashes1 = []
        hashes2 = []
        collect_subtree_hashes(ast1, hashes1, max_depth=3)
        collect_subtree_hashes(ast2, hashes2, max_depth=3)

        common = len(set(hashes1) & set(hashes2))
        total = len(set(hashes1) | set(hashes2))

        return common / total if total > 0 else 0.0

    def tree_depth_similarity(self, ast1: ASTNode, ast2: ASTNode) -> float:

        def max_depth(node: ASTNode) -> int:
            if not node.children:
                return 1
            return 1 + max((max_depth(child) for child in node.children), default=0)

        depth1 = max_depth(ast1)
        depth2 = max_depth(ast2)

        if depth1 == 0 and depth2 == 0:
            return 1.0

        return 1 - abs(depth1 - depth2) / max(depth1, depth2)

    def get_tree_statistics(self, node: ASTNode) -> Dict[str, Any]:
        stats = {
            'total_nodes': 0,
            'node_types': {},
            'max_depth': 0,
            'avg_children_per_node': 0
        }

        def traverse(node: ASTNode, depth: int):
            stats['total_nodes'] += 1
            stats['node_types'][node.node_type] = stats['node_types'].get(node.node_type, 0) + 1
            stats['max_depth'] = max(stats['max_depth'], depth)

            for child in node.children:
                traverse(child, depth + 1)

        if node:
            traverse(node, 1)

            # Average number of children
            if stats['total_nodes'] > 1:
                leaf_nodes = sum(1 for node_type, count in stats['node_types'].items()
                                 if node_type in ['Name', 'Constant', 'Num', 'Str'])
                non_leaf_nodes = stats['total_nodes'] - leaf_nodes
                if non_leaf_nodes > 0:
                    total_children = stats['total_nodes'] - 1  # All nodes except root
                    stats['avg_children_per_node'] = total_children / non_leaf_nodes

        return stats

    def find_similar_nodes(self, ast1: ASTNode, ast2: ASTNode, threshold: float = 0.8) -> List[Dict]:
        similar_nodes = []

        def compare_nodes(node1: ASTNode, node2: ASTNode, path1: str = "", path2: str = ""):
            type_match = 1.0 if node1.node_type == node2.node_type else 0.0
            value_match = 1.0 if node1.value == node2.value else 0.0

            child_hashes1 = [child.structural_hash() for child in node1.children]
            child_hashes2 = [child.structural_hash() for child in node2.children]

            common_children = len(set(child_hashes1) & set(child_hashes2))
            total_children = len(set(child_hashes1) | set(child_hashes2))
            child_similarity = common_children / total_children if total_children > 0 else 1.0

            similarity = (type_match * 0.4 + value_match * 0.2 + child_similarity * 0.4)

            if similarity >= threshold:
                similar_nodes.append({
                    'node1_type': node1.node_type,
                    'node1_value': node1.value,
                    'node1_line': node1.line,
                    'node2_type': node2.node_type,
                    'node2_value': node2.value,
                    'node2_line': node2.line,
                    'similarity': similarity,
                    'path1': path1,
                    'path2': path2
                })

            for i, child1 in enumerate(node1.children):
                for j, child2 in enumerate(node2.children):
                    compare_nodes(child1, child2,
                                  f"{path1}/{i}",
                                  f"{path2}/{j}")

        compare_nodes(ast1, ast2, "root", "root")
        return similar_nodes

    def integrate_results(self, phase1_results: Dict, ast_results: Dict,
                          code1: str, code2: str) -> Dict[str, Any]:
        token_score = phase1_results.get('overall_similarity', 0) / 100  # Convert to 0-1
        ast_score = ast_results.get('ast_similarity_score', 0) / 100  # Convert to 0-1

        weights = self.config.get('integration_weights', {'token': 0.3, 'ast': 0.7})

        combined_score = (weights['token'] * token_score + weights['ast'] * ast_score) * 100

        is_plagiarism = combined_score >= (self.config['plagiarism_threshold'] * 100)

        token_matches = phase1_results.get('matched_sections', [])
        token_metrics = phase1_results.get('similarity_metrics', {})

        return {
            'phase_integration': 'combined_phase1_phase2',
            'combined_similarity_score': combined_score,
            'token_similarity_score': token_score * 100,
            'ast_similarity_score': ast_score * 100,
            'final_decision': 'PLAGIARISM_SUSPECTED' if is_plagiarism else 'CLEAN',
            'is_plagiarism_suspected': is_plagiarism,
            'weights_applied': weights,
            'threshold': self.config['plagiarism_threshold'],

            'phase1_details': {
                'overall_similarity': phase1_results.get('overall_similarity', 0),
                'token_metrics': token_metrics,
                'token_counts': phase1_results.get('token_counts', {}),
                'matched_sections_count': len(token_matches)
            },

            'phase2_details': {
                'ast_similarity_metrics': ast_results.get('ast_similarity_metrics', {}),
                'ast_statistics': ast_results.get('ast_statistics', {}),
                'matched_nodes_count': ast_results.get('matched_nodes_count', 0)
            },

            'input_info': {
                'language': self.language,
                'code1_length': len(code1),
                'code2_length': len(code2)
            }
        }


class Phase2ASTSimilarity:

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.analyzer = None

    def analyze_code_pair(self, code1: str, code2: str, language: str = 'python',
                          phase1_results: Dict = None) -> Dict[str, Any]:
        self.analyzer = ASTSimilarityAnalyzer(language, self.config_path)

        if phase1_results:
            return self.analyzer.analyze_with_phase1(code1, code2, phase1_results)
        else:
            ast1 = self.analyzer.parse_code(code1)
            ast2 = self.analyzer.parse_code(code2)
            return self.analyzer.calculate_ast_similarity(ast1, ast2)

    def analyze_files(self, file1: str, file2: str, language: str = 'auto') -> Dict[str, Any]:
        if language == 'auto':
            lang = self.detect_language_from_file(file1)
        else:
            lang = language

        with open(file1, 'r', encoding='utf-8') as f:
            code1 = f.read()

        with open(file2, 'r', encoding='utf-8') as f:
            code2 = f.read()

        return self.analyze_code_pair(code1, code2, lang)

    def detect_language_from_file(self, filename: str) -> str:
        ext = Path(filename).suffix.lower()

        extensions_map = {
            '.py': 'python',
            '.java': 'java',
            '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', '.hpp': 'cpp', '.h': 'cpp',
        }

        return extensions_map.get(ext, 'python')  # Default to Python

    def analyze_code_pair(self, code1: str, code2: str, language: str = 'python',
                          phase1_results: Dict = None) -> Dict[str, Any]:
        self.analyzer = ASTSimilarityAnalyzer(language, self.config_path)

        ast1 = self.analyzer.parse_code(code1)
        ast2 = self.analyzer.parse_code(code2)

        if phase1_results:
            results = self.analyzer.analyze_with_phase1(code1, code2, phase1_results)
        else:
            results = self.analyzer.calculate_ast_similarity(ast1, ast2)

        if hasattr(ast1, 'to_dict'):
            results['ast1_dict'] = ast1.to_dict()
        if hasattr(ast2, 'to_dict'):
            results['ast2_dict'] = ast2.to_dict()

        return results