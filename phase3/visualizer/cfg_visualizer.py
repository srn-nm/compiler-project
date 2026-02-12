"""
CFG Visualization Module
"""

from typing import Dict, List, Any, Optional
import json

from ..analyzer.cfg_builder import ControlFlowGraph, CFGNode, NodeType


def visualize_cfg(cfg: ControlFlowGraph, max_nodes: int = 20) -> str:
    """Visualize CFG as text"""
    if not cfg.nodes:
        return "Empty CFG"

    result = []
    result.append("=" * 70)
    result.append("Control Flow Graph Visualization")
    result.append("=" * 70)

    # Basic info
    result.append(f"\nBasic Information:")
    result.append(f"  • Total Nodes: {len(cfg.nodes)}")
    result.append(f"  • Total Edges: {len(cfg.edges)}")
    result.append(f"  • Entry Node: {cfg.entry_node_id}")
    result.append(f"  • Exit Node: {cfg.exit_node_id}")
    result.append(f"  • Cyclomatic Complexity: {cfg.get_cyclomatic_complexity()}")
    result.append(f"\nNodes (showing first {max_nodes}):")
    displayed_count = 0
    for node_id, node in cfg.nodes.items():
        if displayed_count >= max_nodes:
            result.append(f"  ... and {len(cfg.nodes) - max_nodes} more nodes")
            break

        node_info = f"  [{node_id}] {node.type.value}: {node.label}"
        if node.line_start:
            node_info += f" (lines {node.line_start}-{node.line_end or node.line_start})"
        if node.statements:
            stmt_preview = ' '.join(node.statements[:3])
            if len(node.statements) > 3:
                stmt_preview += "..."
            node_info += f" - {stmt_preview}"

        result.append(node_info)
        displayed_count += 1

    result.append(f"\nEdges:")
    for from_id, to_id, edge_data in cfg.edges[:max_nodes]:
        label = edge_data.get('label', '')
        result.append(f"  {from_id} -> {to_id}" + (f" [{label}]" if label else ""))

    if len(cfg.edges) > max_nodes:
        result.append(f"  ... and {len(cfg.edges) - max_nodes} more edges")

    result.append(f"\nSample Execution Paths:")
    paths = cfg.get_execution_paths(max_paths=3)
    for i, path in enumerate(paths, 1):
        result.append(f"  Path {i}: {' -> '.join(map(str, path))}")

    result.append("=" * 70)
    return "\n".join(result)


def generate_cfg_report(results: Dict[str, Any]) -> str:
    """Generate CFG analysis report"""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("CFG Analysis Report")
    report_lines.append("=" * 70)

    if 'combined_similarity_score' in results:
        score = results['combined_similarity_score']
        report_lines.append(f"\nCombined Similarity Score: {score:.2f}%")
    else:
        score = results.get('cfg_similarity_score', 0)
        report_lines.append(f"\nCFG Similarity Score: {score:.2f}%")

    is_suspected = results.get('is_plagiarism_suspected', False)
    threshold = results.get('threshold_used', 0.7) * 100

    report_lines.append(f"Detection Threshold: {threshold:.0f}%")

    if is_suspected:
        report_lines.append(f"Result: Similar (Possible Plagiarism)")
    else:
        report_lines.append(f"Result: Not Similar")

    if 'cfg_statistics' in results:
        stats = results['cfg_statistics']
        report_lines.append("\nCFG Statistics:")

        if 'code1' in stats:
            s1 = stats['code1']
            report_lines.append(f"  Code 1:")
            report_lines.append(f"    • Nodes: {s1.get('node_count', 0)}")
            report_lines.append(f"    • Edges: {s1.get('edge_count', 0)}")
            if 'control_structures' in s1:
                cs = s1['control_structures']
                report_lines.append(f"    • Decisions: {cs.get('decisions', 0)}")
                report_lines.append(f"    • Loops: {cs.get('loops', 0)}")

        if 'code2' in stats:
            s2 = stats['code2']
            report_lines.append(f"  Code 2:")
            report_lines.append(f"    • Nodes: {s2.get('node_count', 0)}")
            report_lines.append(f"    • Edges: {s2.get('edge_count', 0)}")
            if 'control_structures' in s2:
                cs = s2['control_structures']
                report_lines.append(f"    • Decisions: {cs.get('decisions', 0)}")
                report_lines.append(f"    • Loops: {cs.get('loops', 0)}")

    if 'cfg_similarity_metrics' in results:
        metrics = results['cfg_similarity_metrics']
        report_lines.append("\nSimilarity Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                report_lines.append(f"  • {key}: {value:.3f}")

    report_lines.append("\n" + "=" * 70)
    return "\n".join(report_lines)


def generate_cfg_dot_file(cfg: ControlFlowGraph, filename: str) -> None:
    """Generate DOT file for graph visualization"""
    try:
        import pydot
    except ImportError:
        print("pydot is not installed. Install it with: pip install pydot")
        return

    graph = pydot.Dot(graph_type='digraph', rankdir='TB')

    for node_id, node in cfg.nodes.items():
        label = f"{node.type.value}\\n{node.label}"
        if node.line_start:
            label += f"\\nLines: {node.line_start}"
            if node.line_end and node.line_end != node.line_start:
                label += f"-{node.line_end}"

        color = get_node_color(node.type)
        shape = get_node_shape(node.type)

        graph.add_node(pydot.Node(
            str(node_id),
            label=label,
            shape=shape,
            style='filled',
            fillcolor=color,
            fontname='Arial'
        ))

    for from_id, to_id, edge_data in cfg.edges:
        label = edge_data.get('label', '')
        graph.add_edge(pydot.Edge(
            str(from_id),
            str(to_id),
            label=label,
            fontname='Arial'
        ))

    graph.write(filename, format='raw')
    print(f"DOT file saved: {filename}")


def get_node_color(node_type: NodeType) -> str:
    """Get color for node type"""
    colors = {
        NodeType.ENTRY: '#90EE90',      # Light green
        NodeType.EXIT: '#FFB6C1',       # Light pink
        NodeType.BASIC_BLOCK: '#E0FFFF',  # Light cyan
        NodeType.DECISION: '#FFD700',   # Gold
        NodeType.LOOP: '#FFA07A',       # Light salmon
        NodeType.FUNCTION: '#DDA0DD',   # Plum
        NodeType.RETURN: '#87CEEB',     # Sky blue
        NodeType.CALL: '#F0E68C'        # Khaki
    }
    return colors.get(node_type, '#FFFFFF')


def get_node_shape(node_type: NodeType) -> str:
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