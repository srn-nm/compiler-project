"""
Building Control Flow Graph (CFG) from AST
"""

from typing import Dict, List, Set, Any, Optional, Tuple
from enum import Enum
import networkx as nx


class NodeType(Enum):
    """Types of CFG nodes"""
    ENTRY = "entry"
    EXIT = "exit"
    BASIC_BLOCK = "basic_block"
    DECISION = "decision"
    LOOP = "loop"
    FUNCTION = "function"
    RETURN = "return"
    CALL = "call"


class CFGNode:
    """Class representing a node in CFG"""
    def __init__(self,
                 node_id: int,
                 node_type: NodeType,
                 label: str = "",
                 line_start: int = None,
                 line_end: int = None,
                 statements: List[str] = None,
                 condition: str = None):
        self.id = node_id
        self.type = node_type
        self.label = label or str(node_type.value)
        self.line_start = line_start
        self.line_end = line_end
        self.statements = statements if statements is not None else []
        self.condition = condition

        self.dominators: Set[int] = set()
        self.post_dominators: Set[int] = set()
        self.in_edges: List[int] = []
        self.out_edges: List[int] = []

    def add_statement(self, statement: str, line: int = None):
        """Add statement to block"""
        if line and not self.line_start:
            self.line_start = line
        if line and (not self.line_end or line > self.line_end):
            self.line_end = line

        self.statements.append(statement)

    def __repr__(self) -> str:
        return f"CFGNode(id={self.id}, type={self.type.value}, label='{self.label}')"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'type': self.type.value,
            'label': self.label,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'statement_count': len(self.statements),
            'condition': self.condition
        }


class BasicBlock:
    """Basic block - sequence of instructions without branching"""

    def __init__(self, block_id: int):
        self.id = block_id
        self.statements: List[Dict[str, Any]] = []
        self.entry_node: Optional[CFGNode] = None
        self.exit_node: Optional[CFGNode] = None
        self.predecessors: Set[int] = set()
        self.successors: Set[int] = set()

    def add_statement(self, statement: str, line: int, node_type: str = "statement"):
        """Add statement to block"""
        self.statements.append({
            'text': statement,
            'line': line,
            'type': node_type
        })

    def is_empty(self) -> bool:
        """Check if block is empty"""
        return len(self.statements) == 0


class CFGBuilder:
    """Class for building CFG from AST"""

    def __init__(self, language: str = 'python'):
        self.language = language
        self.node_counter = 0
        self.nodes: Dict[int, CFGNode] = {}
        self.edges: List[Tuple[int, int, Dict]] = []
        self.basic_blocks: Dict[int, BasicBlock] = {}
        self.function_cfgs: Dict[str, Dict] = {}

    def _get_new_node_id(self) -> int:
        """Get new node ID"""
        self.node_counter += 1
        return self.node_counter

    def build_from_ast(self, ast_data: Dict) -> Dict[str, Any]:
        """
        Build CFG from AST

        """
        entry_node = self._create_node(NodeType.ENTRY, "entry")
        exit_node = self._create_node(NodeType.EXIT, "exit")

        cfg_data = {
            'entry_id': entry_node.id,
            'exit_id': exit_node.id,
            'nodes': self.nodes,
            'edges': self.edges,
            'basic_blocks': self.basic_blocks
        }

        self._process_ast_node(ast_data, entry_node.id, exit_node.id)

        self._connect_to_exit(exit_node.id)

        self._calculate_graph_properties(cfg_data)

        return cfg_data

    def _create_node(self, node_type: NodeType, label: str = "", **kwargs) -> CFGNode:
        """Create new node"""
        node_id = self._get_new_node_id()
        node = CFGNode(node_id, node_type, label, **kwargs)
        self.nodes[node_id] = node
        return node

    def _add_edge(self, from_id: int, to_id: int, label: str = "", **kwargs):
        """Add edge between two nodes"""
        edge_data = {'label': label, **kwargs}
        self.edges.append((from_id, to_id, edge_data))

        if from_id in self.nodes:
            self.nodes[from_id].out_edges.append(to_id)
        if to_id in self.nodes:
            self.nodes[to_id].in_edges.append(from_id)

    def _process_ast_node(self, ast_node: Dict, current_node_id: int, exit_node_id: int) -> int:
        """
        Process an AST node and build corresponding CFG

        """
        if not ast_node:
            return current_node_id

        node_type = ast_node.get('type', '')

        if node_type == 'Module':
            return self._process_module(ast_node, current_node_id, exit_node_id)
        elif node_type == 'FunctionDef':
            return self._process_function(ast_node, current_node_id, exit_node_id)
        elif node_type == 'ClassDef':
            return self._process_class(ast_node, current_node_id, exit_node_id)
        elif node_type == 'If':
            return self._process_if_statement(ast_node, current_node_id, exit_node_id)
        elif node_type == 'For':
            return self._process_for_loop(ast_node, current_node_id, exit_node_id)
        elif node_type == 'While':
            return self._process_while_loop(ast_node, current_node_id, exit_node_id)
        elif node_type == 'Return':
            return self._process_return(ast_node, current_node_id, exit_node_id)
        elif node_type in ['Assign', 'AugAssign', 'AnnAssign']:
            return self._process_assignment(ast_node, current_node_id, exit_node_id)
        elif node_type == 'Expr':
            return self._process_expression(ast_node, current_node_id, exit_node_id)
        else:
            last_node_id = current_node_id
            for child in ast_node.get('children', []):
                last_node_id = self._process_ast_node(child, last_node_id, exit_node_id)
            return last_node_id

    def _process_module(self, module_node: Dict, current_node_id: int, exit_node_id: int) -> int:
        """Process module (top-level)"""
        last_node_id = current_node_id
        for child in module_node.get('children', []):
            last_node_id = self._process_ast_node(child, last_node_id, exit_node_id)
        return last_node_id

    def _process_function(self, func_node: Dict, current_node_id: int, exit_node_id: int) -> int:
        """Process function"""
        func_name = func_node.get('value', 'unnamed_function')

        func_entry = self._create_node(NodeType.FUNCTION, f"func_{func_name}")
        self._add_edge(current_node_id, func_entry.id, label="calls")

        last_node_id = func_entry.id
        for child in func_node.get('children', []):
            child_type = child.get('type', '')
            if child_type == 'arguments':
                continue
            last_node_id = self._process_ast_node(child, last_node_id, exit_node_id)

        if last_node_id != exit_node_id:
            self._add_edge(last_node_id, exit_node_id, label="function_end")

        return current_node_id

    def _process_if_statement(self, if_node: Dict, current_node_id: int, exit_node_id: int) -> int:
        """Process if conditional statement"""
        condition_text = self._extract_condition(if_node)
        decision_node = self._create_node(NodeType.DECISION, f"if_{condition_text}")
        self._add_edge(current_node_id, decision_node.id, label="if")

        then_block_id = None
        else_block_id = None

        for child in if_node.get('children', []):
            child_type = child.get('type', '')
            if child_type == 'test':
                continue
            elif child_type == 'body':
                then_start = self._create_node(NodeType.BASIC_BLOCK, "then_block")
                self._add_edge(decision_node.id, then_start.id, label="true")

                last_then_id = then_start.id
                for stmt in child.get('children', []):
                    last_then_id = self._process_ast_node(stmt, last_then_id, exit_node_id)

                then_block_id = last_then_id

            elif child_type == 'orelse':
                else_start = self._create_node(NodeType.BASIC_BLOCK, "else_block")
                self._add_edge(decision_node.id, else_start.id, label="false")

                last_else_id = else_start.id
                for stmt in child.get('children', []):
                    last_else_id = self._process_ast_node(stmt, last_else_id, exit_node_id)

                else_block_id = last_else_id

        merge_node = self._create_node(NodeType.BASIC_BLOCK, "merge")

        if then_block_id:
            self._add_edge(then_block_id, merge_node.id, label="then_end")
        if else_block_id:
            self._add_edge(else_block_id, merge_node.id, label="else_end")
        else:
            self._add_edge(decision_node.id, merge_node.id, label="false_to_merge")

        return merge_node.id

    def _process_for_loop(self, for_node: Dict, current_node_id: int, exit_node_id: int) -> int:
        """Process for loop"""
        loop_entry = self._create_node(NodeType.LOOP, "for_loop")
        self._add_edge(current_node_id, loop_entry.id, label="enters_loop")

        body_start = self._create_node(NodeType.BASIC_BLOCK, "loop_body")
        self._add_edge(loop_entry.id, body_start.id, label="loop_start")

        last_body_id = body_start.id
        for child in for_node.get('children', []):
            child_type = child.get('type', '')
            if child_type == 'target' or child_type == 'iter':
                continue
            elif child_type == 'body':
                for stmt in child.get('children', []):
                    last_body_id = self._process_ast_node(stmt, last_body_id, exit_node_id)

        self._add_edge(last_body_id, loop_entry.id, label="loop_back")
        loop_exit = self._create_node(NodeType.BASIC_BLOCK, "loop_exit")
        self._add_edge(loop_entry.id, loop_exit.id, label="exits_loop")

        return loop_exit.id

    def _process_while_loop(self, while_node: Dict, current_node_id: int, exit_node_id: int) -> int:
        """Process while loop"""
        condition_text = self._extract_condition(while_node)
        loop_condition = self._create_node(NodeType.DECISION, f"while_{condition_text}")
        self._add_edge(current_node_id, loop_condition.id, label="enters_while")

        body_start = self._create_node(NodeType.BASIC_BLOCK, "while_body")
        self._add_edge(loop_condition.id, body_start.id, label="true")

        last_body_id = body_start.id
        for child in while_node.get('children', []):
            child_type = child.get('type', '')
            if child_type == 'test':
                continue
            elif child_type == 'body':
                for stmt in child.get('children', []):
                    last_body_id = self._process_ast_node(stmt, last_body_id, exit_node_id)

        self._add_edge(last_body_id, loop_condition.id, label="loop_back")

        loop_exit = self._create_node(NodeType.BASIC_BLOCK, "while_exit")
        self._add_edge(loop_condition.id, loop_exit.id, label="false")

        return loop_exit.id

    def _process_return(self, return_node: Dict, current_node_id: int, exit_node_id: int) -> int:
        """Process return statement"""
        return_node_cfg = self._create_node(NodeType.RETURN, "return")
        self._add_edge(current_node_id, return_node_cfg.id, label="returns")
        self._add_edge(return_node_cfg.id, exit_node_id, label="exit")
        return return_node_cfg.id

    def _process_assignment(self, assign_node: Dict, current_node_id: int, exit_node_id: int) -> int:
        """Process assignment"""
        assign_text = self._extract_statement_text(assign_node)
        assign_cfg = self._create_node(
            NodeType.BASIC_BLOCK,
            "assignment",
            statements=[assign_text],
            line_start=assign_node.get('line')
        )
        self._add_edge(current_node_id, assign_cfg.id, label="assigns")
        return assign_cfg.id

    def _process_expression(self, expr_node: Dict, current_node_id: int, exit_node_id: int) -> int:
        """Process expression"""
        expr_text = self._extract_statement_text(expr_node)
        expr_cfg = self._create_node(
            NodeType.BASIC_BLOCK,
            "expression",
            statements=[expr_text],
            line_start=expr_node.get('line')
        )
        self._add_edge(current_node_id, expr_cfg.id, label="executes")
        return expr_cfg.id

    def _extract_condition(self, node: Dict) -> str:
        """Extract condition text"""
        for child in node.get('children', []):
            if child.get('type') == 'test':
                return self._extract_statement_text(child)
        return "condition"

    def _extract_statement_text(self, node: Dict) -> str:
        """Extract statement text"""
        node_type = node.get('type', '')
        value = node.get('value', '')

        if value:
            return f"{node_type}: {value}"
        else:
            return node_type

    def _connect_to_exit(self, exit_node_id: int):
        """Connect nodes without output to exit"""
        for node_id, node in self.nodes.items():
            if node.type != NodeType.EXIT and not node.out_edges:
                # Found node without output
                self._add_edge(node_id, exit_node_id, label="implicit_exit")

    def _calculate_graph_properties(self, cfg_data: Dict):
        """Calculate CFG graph properties"""
        G = nx.DiGraph()

        for node_id, node in self.nodes.items():
            G.add_node(node_id, **node.to_dict())

        for from_id, to_id, edge_data in self.edges:
            G.add_edge(from_id, to_id, **edge_data)

        for node_id in G.nodes():
            node = self.nodes.get(node_id)
            if node:
                node.in_edges = list(G.predecessors(node_id))
                node.out_edges = list(G.successors(node_id))

        self._identify_basic_blocks(G)

        cfg_data['graph'] = G
        cfg_data['graph_properties'] = {
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'is_dag': nx.is_directed_acyclic_graph(G),
            'strongly_connected_components': list(nx.strongly_connected_components(G)),
            'entry_node': cfg_data['entry_id'],
            'exit_node': cfg_data['exit_id']
        }

    def _identify_basic_blocks(self, G: nx.DiGraph):
        """Identify basic blocks in CFG"""
        visited = set()
        block_id = 0

        for node_id in G.nodes():
            if node_id in visited:
                continue

            block = BasicBlock(block_id)
            current_node_id = node_id

            while (current_node_id and
                   current_node_id not in visited and
                   G.out_degree(current_node_id) <= 1 and
                   G.in_degree(current_node_id) <= 1):

                node_data = G.nodes[current_node_id]
                if 'line_start' in node_data:
                    block.add_statement(
                        f"Node {current_node_id}: {node_data.get('label', '')}",
                        node_data.get('line_start', 0),
                        node_data.get('type', 'statement')
                    )

                visited.add(current_node_id)
                successors = list(G.successors(current_node_id))
                if len(successors) == 1:
                    current_node_id = successors[0]
                else:
                    break

            if not block.is_empty():
                self.basic_blocks[block_id] = block
                block_id += 1


class ControlFlowGraph:
    """Main Control Flow Graph class"""

    def __init__(self, language: str = 'python'):
        self.language = language
        self.nodes: Dict[int, CFGNode] = {}
        self.edges: List[Tuple[int, int, Dict]] = []
        self.entry_node_id: Optional[int] = None
        self.exit_node_id: Optional[int] = None
        self.basic_blocks: Dict[int, BasicBlock] = {}
        self.dominator_tree: Dict[int, Set[int]] = {}
        self.post_dominator_tree: Dict[int, Set[int]] = {}
        self.control_dependencies: Dict[int, Set[int]] = {}

    def add_node(self, node: CFGNode) -> int:
        """Add node to CFG"""
        self.nodes[node.id] = node
        return node.id

    def add_edge(self, from_id: int, to_id: int, label: str = "", **kwargs):
        """Add edge"""
        edge_data = {'label': label, **kwargs}
        self.edges.append((from_id, to_id, edge_data))

        if from_id in self.nodes:
            self.nodes[from_id].out_edges.append(to_id)
        if to_id in self.nodes:
            self.nodes[to_id].in_edges.append(from_id)

    def build_from_ast(self, ast_data: Dict) -> 'ControlFlowGraph':
        """Build CFG from AST"""
        builder = CFGBuilder(self.language)
        cfg_data = builder.build_from_ast(ast_data)

        self.nodes = cfg_data['nodes']
        self.edges = cfg_data['edges']
        self.basic_blocks = cfg_data['basic_blocks']
        self.entry_node_id = cfg_data.get('entry_id')
        self.exit_node_id = cfg_data.get('exit_id')
        self._calculate_dominators()
        self._calculate_post_dominators()
        self._calculate_control_dependencies()

        return self

    def _calculate_dominators(self):
        """Calculate dominator tree"""
        if not self.entry_node_id:
            return

        all_nodes = set(self.nodes.keys())

        dominators = {node_id: all_nodes.copy() for node_id in all_nodes}
        dominators[self.entry_node_id] = {self.entry_node_id}

        changed = True
        while changed:
            changed = False
            for node_id in all_nodes:
                if node_id == self.entry_node_id:
                    continue

                preds = self.nodes[node_id].in_edges
                if not preds:
                    new_dom = set()
                else:
                    new_dom = dominators[preds[0]].copy()
                    for pred in preds[1:]:
                        new_dom.intersection_update(dominators[pred])

                new_dom.add(node_id)

                if new_dom != dominators[node_id]:
                    dominators[node_id] = new_dom
                    changed = True

        self.dominator_tree = dominators

    def _calculate_post_dominators(self):
        """Calculate reverse dominator tree"""
        if not self.exit_node_id:
            return

        all_nodes = set(self.nodes.keys())

        reverse_edges = {}
        for from_id, to_id, _ in self.edges:
            if to_id not in reverse_edges:
                reverse_edges[to_id] = []
            reverse_edges[to_id].append(from_id)

        post_dominators = {node_id: all_nodes.copy() for node_id in all_nodes}
        post_dominators[self.exit_node_id] = {self.exit_node_id}

        changed = True
        while changed:
            changed = False
            for node_id in all_nodes:
                if node_id == self.exit_node_id:
                    continue

                succs = reverse_edges.get(node_id, [])
                if not succs:
                    new_pdom = set()
                else:
                    new_pdom = post_dominators[succs[0]].copy()
                    for succ in succs[1:]:
                        new_pdom.intersection_update(post_dominators[succ])

                new_pdom.add(node_id)

                if new_pdom != post_dominators[node_id]:
                    post_dominators[node_id] = new_pdom
                    changed = True

        self.post_dominator_tree = post_dominators

    def _calculate_control_dependencies(self):
        """Calculate control dependencies"""
        self.control_dependencies = {node_id: set() for node_id in self.nodes.keys()}

        for from_id, to_id, _ in self.edges:
            from_node = self.nodes[from_id]
            to_node = self.nodes[to_id]

            if from_node.type == NodeType.DECISION:
                self._add_control_dependency(from_id, to_id)

    def _add_control_dependency(self, control_node_id: int, dependent_node_id: int):
        """Add control dependency"""
        if dependent_node_id in self.nodes:
            self.control_dependencies[dependent_node_id].add(control_node_id)

            for child_id in self.nodes[dependent_node_id].out_edges:
                if child_id != control_node_id:
                    self._add_control_dependency(control_node_id, child_id)

    def get_execution_paths(self, max_paths: int = 100) -> List[List[int]]:
        """Find possible execution paths"""
        if not self.entry_node_id:
            return []

        paths = []
        stack = [(self.entry_node_id, [self.entry_node_id])]

        while stack and len(paths) < max_paths:
            current_node, current_path = stack.pop()

            if current_node == self.exit_node_id:
                paths.append(current_path)
                continue

            for next_node in self.nodes[current_node].out_edges:
                if next_node not in current_path:  # Prevent loops
                    stack.append((next_node, current_path + [next_node]))
                elif next_node == current_node:
                    if current_path.count(next_node) < 2:
                        stack.append((next_node, current_path + [next_node]))

        return paths

    def get_cyclomatic_complexity(self) -> int:
        """Calculate cyclomatic complexity"""
        # Formula: E - N + 2P
        E = len(self.edges)
        N = len(self.nodes)
        P = 1

        return E - N + (2 * P)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'edges': [
                {
                    'from': from_id,
                    'to': to_id,
                    'label': edge_data.get('label', '')
                }
                for from_id, to_id, edge_data in self.edges
            ],
            'entry_node': self.entry_node_id,
            'exit_node': self.exit_node_id,
            'basic_blocks': {
                bid: {
                    'id': block.id,
                    'statement_count': len(block.statements),
                    'predecessors': list(block.predecessors),
                    'successors': list(block.successors)
                }
                for bid, block in self.basic_blocks.items()
            },
            'metrics': {
                'node_count': len(self.nodes),
                'edge_count': len(self.edges),
                'cyclomatic_complexity': self.get_cyclomatic_complexity(),
                'control_structures': self._count_control_structures()
            }
        }

    def _count_control_structures(self) -> Dict[str, int]:
        """Count control structures"""
        counts = {
            'decisions': 0,
            'loops': 0,
            'functions': 0,
            'returns': 0,
            'calls': 0
        }

        for node in self.nodes.values():
            if node.type == NodeType.DECISION:
                counts['decisions'] += 1
            elif node.type == NodeType.LOOP:
                counts['loops'] += 1
            elif node.type == NodeType.FUNCTION:
                counts['functions'] += 1
            elif node.type == NodeType.RETURN:
                counts['returns'] += 1
            elif node.type == NodeType.CALL:
                counts['calls'] += 1

        return counts