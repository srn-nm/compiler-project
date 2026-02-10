"""
Phase 2 Results Visualizer with Hierarchical Tree Display
"""

from typing import Dict, Any, List
from .analyzer import ASTNode


class ASTVisualizer:
    """AST visualization class with tree hierarchy display"""
    
    def __init__(self, indent_size: int = 4, max_depth: int = 10):
        self.indent_size = indent_size
        self.max_depth = max_depth
    
    def visualize_ast_tree(self, ast_node: ASTNode, current_depth: int = 0) -> str:
        """
        Display hierarchical AST as a tree

        """
        if current_depth > self.max_depth:
            return "  " * current_depth + "... (depth exceeded)\n"
        
        result = []
        indent = "  " * current_depth

        node_info = f"{indent}├─ [{ast_node.node_type}]"
        if ast_node.value:
            node_info += f" : {ast_node.value}"
        if ast_node.line:
            node_info += f" (line {ast_node.line})"
        
        result.append(node_info + "\n")

        for i, child in enumerate(ast_node.children):
            if i == len(ast_node.children) - 1:
                child_indent = indent + "  └─ "
                result.append(child_indent[:-3] + self.visualize_ast_tree(child, current_depth + 1)[len(indent)+4:])
            else:
                child_indent = indent + "  ├─ "
                result.append(child_indent[:-3] + self.visualize_ast_tree(child, current_depth + 1)[len(indent)+4:])
        
        return "".join(result)
    
    def generate_hierarchy_report(self, ast_node: ASTNode, title: str = "AST Structure") -> str:
        """
        Generate complete hierarchical report

        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append(f"{title}")
        report_lines.append("=" * 70)

        stats = self._calculate_ast_statistics(ast_node)
        report_lines.append(f"\nGeneral Statistics:")
        report_lines.append(f"  • Total Nodes: {stats['total_nodes']}")
        report_lines.append(f"  • Tree Depth: {stats['max_depth']}")
        report_lines.append(f"  • Node Types: {stats['node_types_count']}")

        report_lines.append(f"\nStructural Hierarchy:")
        tree_view = self.visualize_ast_tree(ast_node)
        report_lines.append(tree_view)

        report_lines.append(f"\nKey Elements Identified:")

        classes = self._extract_elements(ast_node, 'ClassDef')
        if classes:
            report_lines.append(f"  • Classes ({len(classes)}):")
            for cls in classes[:5]:
                report_lines.append(f"    - {cls.get('name', 'unnamed')} (line {cls.get('line', '?')})")

        functions = self._extract_elements(ast_node, 'FunctionDef')
        if functions:
            report_lines.append(f"  • Functions ({len(functions)}):")
            for func in functions[:10]:
                report_lines.append(f"    - {func.get('name', 'unnamed')}() (line {func.get('line', '?')})")

        assignments = self._extract_assignments(ast_node)
        if assignments:
            report_lines.append(f"  • Variables ({len(assignments)}):")
            for var in assignments[:15]:
                report_lines.append(f"    - {var.get('name', 'unnamed')} = ... (line {var.get('line', '?')})")
        
        report_lines.append("\n" + "=" * 70)
        return "\n".join(report_lines)
    
    def _calculate_ast_statistics(self, node: ASTNode) -> Dict[str, Any]:
        """Calculate AST statistics"""
        stats = {
            'total_nodes': 0,
            'max_depth': 0,
            'node_types': {},
            'node_types_count': 0
        }
        
        def traverse(n: ASTNode, depth: int):
            stats['total_nodes'] += 1
            stats['max_depth'] = max(stats['max_depth'], depth)
            
            node_type = n.node_type
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
            
            for child in n.children:
                traverse(child, depth + 1)
        
        traverse(node, 1)
        stats['node_types_count'] = len(stats['node_types'])
        return stats
    
    def _extract_elements(self, node: ASTNode, element_type: str) -> List[Dict[str, Any]]:
        """Extract specific elements from AST"""
        elements = []
        
        def extract(n: ASTNode):
            if n.node_type == element_type:
                elem = {'type': n.node_type, 'line': n.line}
                if n.value:
                    elem['name'] = n.value

                if element_type == 'FunctionDef':
                    for child in n.children:
                        if child.node_type == 'arguments':
                            params = []
                            for param_child in child.children:
                                if param_child.node_type == 'arg':
                                    params.append(param_child.value or '?')
                            elem['params'] = params
                            break
                
                elements.append(elem)
            
            for child in n.children:
                extract(child)
        
        extract(node)
        return elements
    
    def _extract_assignments(self, node: ASTNode) -> List[Dict[str, Any]]:
        """Extract defined variables"""
        assignments = []
        
        def extract(n: ASTNode):
            if n.node_type == 'Assign':
                for child in n.children:
                    if child.node_type == 'Name' and child.value:
                        assignments.append({
                            'name': child.value,
                            'line': child.line or n.line,
                            'type': 'assignment'
                        })
                        break
            
            for child in n.children:
                extract(child)
        
        extract(node)
        return assignments
    
    def compare_ast_hierarchies(self, ast1: ASTNode, ast2: ASTNode, 
                              title1: str = "Code 1", 
                              title2: str = "Code 2") -> str:
        """
        Compare two AST structures side by side

        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("Hierarchical Comparison of Two Codes")
        report_lines.append("=" * 70)
        
        # Comparative statistics
        stats1 = self._calculate_ast_statistics(ast1)
        stats2 = self._calculate_ast_statistics(ast2)
        
        report_lines.append(f"\nComparative Statistics:")
        report_lines.append(f"{'':15} {title1:^25} {title2:^25}")
        report_lines.append(f"{'':15} {'─'*25} {'─'*25}")
        report_lines.append(f"Total Nodes:    {stats1['total_nodes']:^25} {stats2['total_nodes']:^25}")
        report_lines.append(f"Tree Depth:    {stats1['max_depth']:^25} {stats2['max_depth']:^25}")
        report_lines.append(f"Node Types:    {stats1['node_types_count']:^25} {stats2['node_types_count']:^25}")

        report_lines.append(f"\nKey Elements Comparison:")

        classes1 = self._extract_elements(ast1, 'ClassDef')
        classes2 = self._extract_elements(ast2, 'ClassDef')
        
        report_lines.append(f"\n🏛️  Classes:")
        report_lines.append(f"  • {title1}: {len(classes1)} classes")
        if classes1:
            report_lines.append(f"    {', '.join([c.get('name', '?') for c in classes1[:3]])}" + 
                              (" ..." if len(classes1) > 3 else ""))
        report_lines.append(f"  • {title2}: {len(classes2)} classes")
        if classes2:
            report_lines.append(f"    {', '.join([c.get('name', '?') for c in classes2[:3]])}" + 
                              (" ..." if len(classes2) > 3 else ""))
        
        # Functions
        funcs1 = self._extract_elements(ast1, 'FunctionDef')
        funcs2 = self._extract_elements(ast2, 'FunctionDef')
        
        report_lines.append(f"\nFunctions:")
        report_lines.append(f"  • {title1}: {len(funcs1)} functions")
        if funcs1:
            func_names1 = [f"{f.get('name', '?')}()" for f in funcs1[:5]]
            report_lines.append(f"    {', '.join(func_names1)}" + 
                              (" ..." if len(funcs1) > 5 else ""))
        report_lines.append(f"  • {title2}: {len(funcs2)} functions")
        if funcs2:
            func_names2 = [f"{f.get('name', '?')}()" for f in funcs2[:5]]
            report_lines.append(f"    {', '.join(func_names2)}" + 
                              (" ..." if len(funcs2) > 5 else ""))
        
        # Find common elements
        common_funcs = self._find_common_elements(funcs1, funcs2)
        if common_funcs:
            report_lines.append(f"\nCommon Functions ({len(common_funcs)}):")
            for func in common_funcs[:10]:
                report_lines.append(f"    • {func}")
        
        # Display sample tree structure
        report_lines.append(f"\nSample Tree Structure of {title1}:")
        sample_tree1 = self._get_sample_tree(ast1, max_nodes=50)
        report_lines.append(sample_tree1)
        
        report_lines.append(f"\nSample Tree Structure of {title2}:")
        sample_tree2 = self._get_sample_tree(ast2, max_nodes=50)
        report_lines.append(sample_tree2)
        
        report_lines.append("\n" + "=" * 70)
        return "\n".join(report_lines)
    
    def _find_common_elements(self, elements1: List[Dict], elements2: List[Dict]) -> List[str]:
        """Find common elements"""
        names1 = {e.get('name', '').lower() for e in elements1 if e.get('name')}
        names2 = {e.get('name', '').lower() for e in elements2 if e.get('name')}
        
        common = names1.intersection(names2)
        return sorted(list(common))
    
    def _get_sample_tree(self, node: ASTNode, max_nodes: int = 20) -> str:
        """Get a sample of tree structure (to prevent large output)"""
        lines = []
        node_count = [0]
        
        def sample_traverse(n: ASTNode, depth: int, prefix: str = ""):
            if node_count[0] >= max_nodes:
                if node_count[0] == max_nodes:
                    lines.append("  " * depth + "... (display limited)")
                    node_count[0] += 1
                return
            
            node_info = f"[{n.node_type}]"
            if n.value and len(n.value) < 20:
                node_info += f" : {n.value}"
            
            lines.append("  " * depth + prefix + node_info)
            node_count[0] += 1
            
            if n.children and node_count[0] < max_nodes:
                for i, child in enumerate(n.children):
                    if node_count[0] >= max_nodes:
                        break
                    child_prefix = "└─ " if i == len(n.children) - 1 else "├─ "
                    sample_traverse(child, depth + 1, child_prefix)
        
        sample_traverse(node, 0)
        return "\n".join(lines)


def generate_phase2_report(results: Dict[str, Any]) -> str:
    """Generate Phase 2 report (compatibility with existing code)"""

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("Phase 2 Analysis Report - Structural Similarity (AST)")
    report_lines.append("=" * 70)
    
    if 'phase_integration' in results:
        report_lines.append(f" Analysis Type: {results['phase_integration']}")
        report_lines.append(f" Combined Score: {results.get('combined_similarity_score', 0):.2f}%")
        report_lines.append(f"  • Token Score: {results.get('token_similarity_score', 0):.2f}%")
        report_lines.append(f"  • Structural Score: {results.get('ast_similarity_score', 0):.2f}%")
    else:
        report_lines.append(f" Structural Score (AST): {results.get('ast_similarity_score', 0):.2f}%")

    threshold = results.get('threshold_used', 0.65) * 100
    is_suspected = results.get('is_plagiarism_suspected', False)
    report_lines.append(f"Detection Threshold: {threshold:.0f}%")
    
    if is_suspected:
        report_lines.append(f"Decision:Similar (Possible Plagiarism)")
    else:
        report_lines.append(f"Decision:Not Similar")
    
    # AST Statistics
    if 'phase2_details' in results and 'ast_statistics' in results['phase2_details']:
        ast_stats = results['phase2_details']['ast_statistics']
        if 'code1' in ast_stats and 'code2' in ast_stats:
            report_lines.append("\nAST Tree Statistics:")
            stats1 = ast_stats['code1']
            stats2 = ast_stats['code2']
            report_lines.append(f"  • Code 1: {stats1.get('total_nodes', 0)} nodes")
            report_lines.append(f"  • Code 2: {stats2.get('total_nodes', 0)} nodes")
    
    report_lines.append("\n" + "=" * 70)
    return "\n".join(report_lines)


def visualize_ast_comparison(results: Dict[str, Any]) -> str:
    return generate_phase2_report(results)