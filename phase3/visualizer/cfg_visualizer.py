"""
CFG Visualization Module - Text and DOT format
"""

from typing import Dict, List, Any, Optional
from phase3.analyzer.cfg_builder import ControlFlowGraph, CFGNode, NodeType


def visualize_cfg(cfg: ControlFlowGraph, max_nodes: int = 20) -> str:
    """Visualize CFG as text tree"""
    if not cfg.nodes:
        return "Empty CFG"
    
    lines = []
    lines.append("=" * 70)
    lines.append("CONTROL FLOW GRAPH")
    lines.append("=" * 70)
    
    # Basic info
    lines.append(f"\n Statistics:")
    lines.append(f"  • Nodes: {len(cfg.nodes)}")
    lines.append(f"  • Edges: {len(cfg.edges)}")
    lines.append(f"  • Entry: {cfg.entry_node_id}")
    lines.append(f"  • Exit:  {cfg.exit_node_id}")
    lines.append(f"  • Cyclomatic Complexity: {cfg.get_cyclomatic_complexity()}")
    
    # Control structures
    controls = cfg.get_control_structures_count()
    lines.append(f"\n Control Structures:")
    lines.append(f"  • Decisions: {controls.get('decisions', 0)}")
    lines.append(f"  • Loops: {controls.get('loops', 0)}")
    lines.append(f"  • Functions: {controls.get('functions', 0)}")
    lines.append(f"  • Returns: {controls.get('returns', 0)}")
    
    # Nodes
    lines.append(f"\n Nodes (first {max_nodes}):")
    displayed = 0
    for node_id, node in cfg.nodes.items():
        if displayed >= max_nodes:
            lines.append(f"  ... and {len(cfg.nodes) - max_nodes} more")
            break
        
        node_info = f"  [{node_id:2d}] {node.type.value:12} - {node.label}"
        if node.line_start:
            node_info += f" (L{node.line_start})"
        if node.statements:
            node_info += f" [{len(node.statements)} stmts]"
        lines.append(node_info)
        displayed += 1
    
    # Edges
    lines.append(f"\n Edges (first {max_nodes}):")
    displayed = 0
    for from_id, to_id, data in cfg.edges:
        if displayed >= max_nodes:
            lines.append(f"  ... and {len(cfg.edges) - max_nodes} more")
            break
        
        label = data.get('label', '')
        edge_str = f"  {from_id:2d} → {to_id:2d}"
        if label:
            edge_str += f" [{label}]"
        lines.append(edge_str)
        displayed += 1
    
    # Sample paths
    lines.append(f"\n  Sample Paths:")
    paths = cfg.get_execution_paths(max_paths=3)
    for i, path in enumerate(paths, 1):
        path_str = " → ".join(map(str, path[:10]))
        if len(path) > 10:
            path_str += " → ..."
        lines.append(f"  Path {i}: {path_str}")
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def generate_cfg_report(results: Dict[str, Any]) -> str:
    """Generate CFG analysis report from results"""
    lines = []
    lines.append("=" * 70)
    lines.append("CFG ANALYSIS REPORT")
    lines.append("=" * 70)
    
    # Score
    if 'combined_similarity_score' in results:
        score = results['combined_similarity_score']
        lines.append(f"\n Combined Score (3 phases): {score:.2f}%")
    else:
        score = results.get('cfg_similarity_score', 0)
        lines.append(f"\n CFG Similarity Score: {score:.2f}%")
    
    # Decision
    is_suspected = results.get('is_plagiarism_suspected', False)
    threshold = results.get('threshold_used', 0.7) * 100
    lines.append(f"\n  Verdict:")
    lines.append(f"  • Threshold: {threshold:.0f}%")
    lines.append(f"  • Result: {' SIMILAR' if is_suspected else ' NOT SIMILAR'}")
    
    # Metrics
    if 'cfg_similarity_metrics' in results:
        metrics = results['cfg_similarity_metrics']
        lines.append(f"\n Similarity Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  • {key:25}: {value*100:6.2f}%")
    
    # Statistics
    if 'cfg_statistics' in results:
        stats = results['cfg_statistics']
        lines.append(f"\n CFG Statistics:")
        
        if 'code1' in stats:
            s1 = stats['code1']
            lines.append(f"  Code 1:")
            lines.append(f"    • Nodes: {s1.get('node_count', 0)}")
            lines.append(f"    • Edges: {s1.get('edge_count', 0)}")
            lines.append(f"    • Complexity: {s1.get('cyclomatic_complexity', 0)}")
        
        if 'code2' in stats:
            s2 = stats['code2']
            lines.append(f"  Code 2:")
            lines.append(f"    • Nodes: {s2.get('node_count', 0)}")
            lines.append(f"    • Edges: {s2.get('edge_count', 0)}")
            lines.append(f"    • Complexity: {s2.get('cyclomatic_complexity', 0)}")
    
    # Similar components
    if 'similar_components' in results and results['similar_components']:
        components = results['similar_components']
        lines.append(f"\n🔍 Similar Components: {len(components)}")
        for i, comp in enumerate(components[:5], 1):
            lines.append(f"  {i}. {comp.get('type', 'unknown')}: "
                        f"{comp.get('similarity', 0)*100:.1f}%")
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def generate_cfg_dot_file(cfg: ControlFlowGraph, filename: str) -> None:
    """Generate DOT file for Graphviz visualization"""
    try:
        import pydot
    except ImportError:
        print(" pydot not installed. Install: pip install pydot")
        return
    
    graph = pydot.Dot(graph_type='digraph', rankdir='TB')
    
    # Add nodes
    for node_id, node in cfg.nodes.items():
        label = f"{node.type.value}\\n{node.label}"
        if node.line_start:
            label += f"\\nL{node.line_start}"
        
        color = _get_node_color(node.type)
        shape = _get_node_shape(node.type)
        
        graph.add_node(pydot.Node(
            str(node_id),
            label=label,
            shape=shape,
            style='filled',
            fillcolor=color,
            fontname='Arial'
        ))
    
    # Add edges
    for from_id, to_id, edge_data in cfg.edges:
        label = edge_data.get('label', '')
        graph.add_edge(pydot.Edge(
            str(from_id),
            str(to_id),
            label=label,
            fontname='Arial'
        ))
    
    # Save
    graph.write(filename, format='raw')
    print(f" DOT file saved: {filename}")


def _get_node_color(node_type: NodeType) -> str:
    """Get color for node type"""
    colors = {
        NodeType.ENTRY: '#90EE90',      
        NodeType.EXIT: '#FFB6C1',       
        NodeType.BASIC_BLOCK: '#E0FFFF',
        NodeType.DECISION: '#FFD700',    
        NodeType.LOOP: '#FFA07A',       
        NodeType.FUNCTION: '#DDA0DD',  
        NodeType.RETURN: '#87CEEB',    
        NodeType.CALL: '#F0E68C'        
    }
    return colors.get(node_type, '#FFFFFF')


def _get_node_shape(node_type: NodeType) -> str:
    """Get shape for node type"""
    shapes = {
        NodeType.ENTRY: 'doublecircle',
        NodeType.EXIT: 'doublecircle',
        NodeType.BASIC_BLOCK: 'box',
        NodeType.DECISION: 'diamond',
        NodeType.LOOP: 'ellipse',
        NodeType.FUNCTION: 'house',
        NodeType.RETURN: 'parallelogram',
        NodeType.CALL: 'component'
    }
    return shapes.get(node_type, 'ellipse')