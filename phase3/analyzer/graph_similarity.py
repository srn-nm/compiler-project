

from typing import Dict, List,Any

from .cfg_builder import ControlFlowGraph, CFGNode, NodeType


class GraphSimilarity:
    """Advanced graph similarity algorithms for CFG comparison"""
    
    def __init__(self):
        self.weights = {
            'structural': 0.35,
            'graph_edit': 0.35,
            'subgraph': 0.30
        }

    def find_similar_components(self, cfg1: ControlFlowGraph,
                               cfg2: ControlFlowGraph) -> List[Dict[str, Any]]:
        """Find similar control structures between two CFGs"""
        similar_components = []
        
        # Find similar decision structures (if/else)
        decisions1 = self._get_nodes_by_type(cfg1, NodeType.DECISION)
        decisions2 = self._get_nodes_by_type(cfg2, NodeType.DECISION)
        
        for d1 in decisions1:
            for d2 in decisions2:
                sim = self._decision_similarity(cfg1, cfg2, d1, d2)
                if sim > 0.7:
                    similar_components.append({
                        'type': 'decision',
                        'node1': d1.id,
                        'node2': d2.id,
                        'similarity': round(sim, 3)
                    })
        
        # Find similar loop structures
        loops1 = self._get_nodes_by_type(cfg1, NodeType.LOOP)
        loops2 = self._get_nodes_by_type(cfg2, NodeType.LOOP)
        
        for l1 in loops1:
            for l2 in loops2:
                sim = self._loop_similarity(cfg1, cfg2, l1, l2)
                if sim > 0.7:
                    similar_components.append({
                        'type': 'loop',
                        'node1': l1.id,
                        'node2': l2.id,
                        'similarity': round(sim, 3)
                    })
        
        # Find similar function structures
        funcs1 = self._get_nodes_by_type(cfg1, NodeType.FUNCTION)
        funcs2 = self._get_nodes_by_type(cfg2, NodeType.FUNCTION)
        
        for f1 in funcs1:
            for f2 in funcs2:
                sim = self._function_similarity(cfg1, cfg2, f1, f2)
                if sim > 0.7:
                    similar_components.append({
                        'type': 'function',
                        'node1': f1.id,
                        'node2': f2.id,
                        'similarity': round(sim, 3)
                    })
        
        return similar_components

    def _decision_similarity(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph,
                            node1: CFGNode, node2: CFGNode) -> float:
        """Calculate similarity between two decision structures"""
        # Get branches
        branches1 = self._extract_branches(cfg1, node1.id)
        branches2 = self._extract_branches(cfg2, node2.id)
        
        if not branches1 and not branches2:
            return 1.0
        
        # Compare number of branches
        if len(branches1) != len(branches2):
            return 0.4
        
        # Compare branch sizes
        similarity = 0.0
        for b1, b2 in zip(branches1, branches2):
            size_sim = 1 - abs(len(b1) - len(b2)) / max(len(b1), len(b2), 1)
            similarity += size_sim
        
        return similarity / len(branches1) if branches1 else 0.0

    def _loop_similarity(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph,
                        node1: CFGNode, node2: CFGNode) -> float:
        """Calculate similarity between two loop structures"""
        body1 = self._extract_loop_body(cfg1, node1.id)
        body2 = self._extract_loop_body(cfg2, node2.id)
        
        if not body1 and not body2:
            return 1.0
        
        # Size similarity
        size_sim = 1 - abs(len(body1) - len(body2)) / max(len(body1), len(body2), 1)
        
        # Nested structure similarity
        nested1 = any(cfg1.nodes[n].type in [NodeType.DECISION, NodeType.LOOP] 
                     for n in body1 if n in cfg1.nodes)
        nested2 = any(cfg2.nodes[n].type in [NodeType.DECISION, NodeType.LOOP]
                     for n in body2 if n in cfg2.nodes)
        
        nested_sim = 1.0 if nested1 == nested2 else 0.5
        
        return size_sim * 0.7 + nested_sim * 0.3

    def _function_similarity(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph,
                            node1: CFGNode, node2: CFGNode) -> float:
        """Calculate similarity between two function structures"""
        body1 = self._extract_function_body(cfg1, node1.id)
        body2 = self._extract_function_body(cfg2, node2.id)
        
        if not body1 and not body2:
            return 1.0
        
        # Size similarity
        size_sim = 1 - abs(len(body1) - len(body2)) / max(len(body1), len(body2), 1)
        
        # Count control structures
        control1 = sum(1 for n in body1 if n in cfg1.nodes and 
                      cfg1.nodes[n].type in [NodeType.DECISION, NodeType.LOOP])
        control2 = sum(1 for n in body2 if n in cfg2.nodes and
                      cfg2.nodes[n].type in [NodeType.DECISION, NodeType.LOOP])
        
        control_sim = 1 - abs(control1 - control2) / max(control1, control2, 1)
        
        return size_sim * 0.6 + control_sim * 0.4

    def _extract_branches(self, cfg: ControlFlowGraph, decision_id: int) -> List[List[int]]:
        """Extract branches from a decision node"""
        branches = []
        node = cfg.nodes.get(decision_id)
        
        if not node or node.type != NodeType.DECISION:
            return branches
        
        for succ_id in node.out_edges:
            branch = self._extract_path(cfg, succ_id, max_depth=5)
            if branch:
                branches.append(branch)
        
        return branches

    def _extract_path(self, cfg: ControlFlowGraph, start_id: int, 
                     max_depth: int = 5) -> List[int]:
        """Extract a path from start node"""
        path = []
        current = start_id
        depth = 0
        
        while current and depth < max_depth:
            path.append(current)
            node = cfg.nodes.get(current)
            
            if not node or not node.out_edges:
                break
            
            current = node.out_edges[0]
            depth += 1
        
        return path

    def _extract_loop_body(self, cfg: ControlFlowGraph, loop_id: int) -> List[int]:
        """Extract loop body nodes"""
        body = []
        visited = set()
        
        def dfs(node_id: int):
            if node_id in visited or node_id == loop_id:
                return
            visited.add(node_id)
            body.append(node_id)
            
            node = cfg.nodes.get(node_id)
            if node:
                for succ in node.out_edges:
                    dfs(succ)
        
        node = cfg.nodes.get(loop_id)
        if node:
            for succ in node.out_edges:
                dfs(succ)
        
        return body

    def _extract_function_body(self, cfg: ControlFlowGraph, func_id: int) -> List[int]:
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
                for succ in node.out_edges:
                    dfs(succ)
        
        node = cfg.nodes.get(func_id)
        if node:
            for succ in node.out_edges:
                dfs(succ)
        
        return body

    def _get_nodes_by_type(self, cfg: ControlFlowGraph, node_type: NodeType) -> List[CFGNode]:
        """Get all nodes of specific type"""
        return [node for node in cfg.nodes.values() if node.type == node_type]