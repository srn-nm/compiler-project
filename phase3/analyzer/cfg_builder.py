"""
Control Flow Graph (CFG) Builder from AST
Complete implementation with all control structures
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
    """Node in Control Flow Graph"""
    
    def __init__(self,
                 node_id: int,
                 node_type: NodeType,
                 label: str = "",
                 line_start: int = None,
                 line_end: int = None,
                 statements: List[str] = None):
        self.id = node_id
        self.type = node_type
        self.label = label or node_type.value
        self.line_start = line_start
        self.line_end = line_end
        self.statements = statements if statements is not None else []
        
        self.in_edges: List[int] = []
        self.out_edges: List[int] = []
        self.dominators: Set[int] = set()
        self.post_dominators: Set[int] = set()

    def add_statement(self, statement: str, line: int = None):
        """Add statement to node"""
        if line:
            if not self.line_start or line < self.line_start:
                self.line_start = line
            if not self.line_end or line > self.line_end:
                self.line_end = line
        self.statements.append(statement)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'type': self.type.value,
            'label': self.label,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'statement_count': len(self.statements),
            'in_degree': len(self.in_edges),
            'out_degree': len(self.out_edges)
        }

    def __repr__(self) -> str:
        return f"CFGNode({self.id}, {self.type.value})"


class BasicBlock:
    """Basic block - sequence of statements without branches"""
    
    def __init__(self, block_id: int):
        self.id = block_id
        self.statements: List[Dict] = []
        self.entry_node: Optional[CFGNode] = None
        self.exit_node: Optional[CFGNode] = None
        self.predecessors: Set[int] = set()
        self.successors: Set[int] = set()

    def add_statement(self, statement: str, line: int, stmt_type: str = "stmt"):
        """Add statement to block"""
        self.statements.append({
            'text': statement,
            'line': line,
            'type': stmt_type
        })

    def is_empty(self) -> bool:
        return len(self.statements) == 0


class CFGBuilder:
    """Build Control Flow Graph from AST"""
    
    def __init__(self, language: str = 'python'):
        self.language = language
        self.node_counter = 0
        self.nodes: Dict[int, CFGNode] = {}
        self.edges: List[Tuple[int, int, Dict]] = []
        self.basic_blocks: Dict[int, BasicBlock] = {}
        self.current_function = None

    def _new_node_id(self) -> int:
        """Generate new node ID"""
        self.node_counter += 1
        return self.node_counter

    def _create_node(self, node_type: NodeType, label: str = "", **kwargs) -> CFGNode:
        """Create new CFG node"""
        node_id = self._new_node_id()
        node = CFGNode(node_id, node_type, label, **kwargs)
        self.nodes[node_id] = node
        return node

    def _add_edge(self, from_id: int, to_id: int, label: str = "", **kwargs):
        """Add edge between nodes"""
        edge_data = {'label': label, **kwargs}
        self.edges.append((from_id, to_id, edge_data))
        
        if from_id in self.nodes:
            self.nodes[from_id].out_edges.append(to_id)
        if to_id in self.nodes:
            self.nodes[to_id].in_edges.append(from_id)

    def build_from_ast(self, ast_data: Dict) -> Dict[str, Any]:
        """Build CFG from AST data"""
        # Reset state
        self.node_counter = 0
        self.nodes = {}
        self.edges = []
        self.basic_blocks = {}
        
        # Create entry and exit nodes
        entry_node = self._create_node(NodeType.ENTRY, "entry")
        exit_node = self._create_node(NodeType.EXIT, "exit")
        
        # Process AST recursively
        last_node = self._process_ast_node(ast_data, entry_node.id, exit_node.id)
        
        # Connect to exit if not already connected
        if last_node != exit_node.id:
            self._add_edge(last_node, exit_node.id, "end")
        
        # Connect all leaf nodes to exit
        self._connect_dangling_nodes(exit_node.id)
        
        return {
            'entry_id': entry_node.id,
            'exit_id': exit_node.id,
            'nodes': self.nodes,
            'edges': self.edges,
            'basic_blocks': self.basic_blocks
        }

    def _process_ast_node(self, ast_node: Dict, current_id: int, exit_id: int) -> int:
        """Process AST node recursively"""
        if not ast_node:
            return current_id

        node_type = ast_node.get('type', '')

        # Handle different node types
        handlers = {
            'Module': self._process_module,
            'FunctionDef': self._process_function,
            'ClassDef': self._process_class,
            'If': self._process_if,
            'For': self._process_for,
            'While': self._process_while,
            'Return': self._process_return,
            'Assign': self._process_assign,
            'AugAssign': self._process_assign,
            'Expr': self._process_expression,
        }

        handler = handlers.get(node_type, self._process_default)
        return handler(ast_node, current_id, exit_id)

    def _process_module(self, node: Dict, current_id: int, exit_id: int) -> int:
        """Process module node"""
        last_id = current_id
        for child in node.get('children', []):
            last_id = self._process_ast_node(child, last_id, exit_id)
        return last_id

    def _process_function(self, node: Dict, current_id: int, exit_id: int) -> int:
        """Process function definition"""
        func_name = node.get('value', 'function')
        
        # Create function entry node
        func_entry = self._create_node(NodeType.FUNCTION, f"func_{func_name}")
        self._add_edge(current_id, func_entry.id, "call")
        
        # Process function body
        last_id = func_entry.id
        for child in node.get('children', []):
            if child.get('type') != 'arguments':
                last_id = self._process_ast_node(child, last_id, exit_id)
        
        # Connect to exit
        if last_id != exit_id:
            self._add_edge(last_id, exit_id, "return")
        
        return current_id

    def _process_class(self, node: Dict, current_id: int, exit_id: int) -> int:
        """Process class definition"""
        # Class doesn't affect control flow directly
        last_id = current_id
        for child in node.get('children', []):
            last_id = self._process_ast_node(child, last_id, exit_id)
        return last_id

    def _process_if(self, node: Dict, current_id: int, exit_id: int) -> int:
        """Process if statement"""
        # Create decision node
        decision = self._create_node(NodeType.DECISION, "if")
        self._add_edge(current_id, decision.id, "condition")
        
        then_id = None
        else_id = None
        
        # Process body and orelse
        for child in node.get('children', []):
            child_type = child.get('type', '')
            
            if child_type == 'body':
                then_start = self._create_node(NodeType.BASIC_BLOCK, "then")
                self._add_edge(decision.id, then_start.id, "true")
                
                last_then = then_start.id
                for stmt in child.get('children', []):
                    last_then = self._process_ast_node(stmt, last_then, exit_id)
                then_id = last_then
                
            elif child_type == 'orelse' and child.get('children'):
                else_start = self._create_node(NodeType.BASIC_BLOCK, "else")
                self._add_edge(decision.id, else_start.id, "false")
                
                last_else = else_start.id
                for stmt in child.get('children', []):
                    last_else = self._process_ast_node(stmt, last_else, exit_id)
                else_id = last_else
        
        # Create merge node
        merge = self._create_node(NodeType.BASIC_BLOCK, "merge")
        
        if then_id:
            self._add_edge(then_id, merge.id, "then_end")
        if else_id:
            self._add_edge(else_id, merge.id, "else_end")
        if not then_id and not else_id:
            self._add_edge(decision.id, merge.id, "no_branch")
        
        return merge.id

    def _process_for(self, node: Dict, current_id: int, exit_id: int) -> int:
        """Process for loop"""
        # Create loop header
        loop_header = self._create_node(NodeType.LOOP, "for_header")
        self._add_edge(current_id, loop_header.id, "enter_loop")
        
        # Process loop body
        body_start = self._create_node(NodeType.BASIC_BLOCK, "for_body")
        self._add_edge(loop_header.id, body_start.id, "loop_start")
        
        last_body = body_start.id
        for child in node.get('children', []):
            if child.get('type') == 'body':
                for stmt in child.get('children', []):
                    last_body = self._process_ast_node(stmt, last_body, exit_id)
        
        # Back edge
        self._add_edge(last_body, loop_header.id, "loop_back")
        
        # Loop exit
        loop_exit = self._create_node(NodeType.BASIC_BLOCK, "for_exit")
        self._add_edge(loop_header.id, loop_exit.id, "exit_loop")
        
        return loop_exit.id

    def _process_while(self, node: Dict, current_id: int, exit_id: int) -> int:
        """Process while loop"""
        # Decision node for condition
        condition = self._create_node(NodeType.DECISION, "while")
        self._add_edge(current_id, condition.id, "enter_loop")
        
        # Loop body
        body_start = self._create_node(NodeType.BASIC_BLOCK, "while_body")
        self._add_edge(condition.id, body_start.id, "true")
        
        last_body = body_start.id
        for child in node.get('children', []):
            if child.get('type') == 'body':
                for stmt in child.get('children', []):
                    last_body = self._process_ast_node(stmt, last_body, exit_id)
        
        # Back edge
        self._add_edge(last_body, condition.id, "loop_back")
        
        # Loop exit
        loop_exit = self._create_node(NodeType.BASIC_BLOCK, "while_exit")
        self._add_edge(condition.id, loop_exit.id, "false")
        
        return loop_exit.id

    def _process_return(self, node: Dict, current_id: int, exit_id: int) -> int:
        """Process return statement"""
        ret_node = self._create_node(NodeType.RETURN, "return")
        self._add_edge(current_id, ret_node.id, "return")
        self._add_edge(ret_node.id, exit_id, "exit")
        return ret_node.id

    def _process_assign(self, node: Dict, current_id: int, exit_id: int) -> int:
        """Process assignment"""
        assign_node = self._create_node(
            NodeType.BASIC_BLOCK,
            "assign",
            line_start=node.get('line')
        )
        assign_node.add_statement("assignment", node.get('line'))
        self._add_edge(current_id, assign_node.id, "assign")
        return assign_node.id

    def _process_expression(self, node: Dict, current_id: int, exit_id: int) -> int:
        """Process expression"""
        expr_node = self._create_node(
            NodeType.BASIC_BLOCK,
            "expr",
            line_start=node.get('line')
        )
        expr_node.add_statement("expression", node.get('line'))
        self._add_edge(current_id, expr_node.id, "exec")
        return expr_node.id

    def _process_default(self, node: Dict, current_id: int, exit_id: int) -> int:
        """Default handler - process children"""
        last_id = current_id
        for child in node.get('children', []):
            last_id = self._process_ast_node(child, last_id, exit_id)
        return last_id

    def _connect_dangling_nodes(self, exit_id: int):
        """Connect nodes with no outgoing edges to exit"""
        for node_id, node in self.nodes.items():
            if node.type not in [NodeType.EXIT, NodeType.RETURN]:
                if not node.out_edges:
                    self._add_edge(node_id, exit_id, "implicit_exit")


class ControlFlowGraph:
    """Main Control Flow Graph class"""
    
    def __init__(self, language: str = 'python'):
        self.language = language
        self.nodes: Dict[int, CFGNode] = {}
        self.edges: List[Tuple[int, int, Dict]] = []
        self.entry_node_id: Optional[int] = None
        self.exit_node_id: Optional[int] = None
        self.basic_blocks: Dict[int, BasicBlock] = {}

    def build_from_ast(self, ast_data: Dict) -> 'ControlFlowGraph':
        """Build CFG from AST"""
        builder = CFGBuilder(self.language)
        cfg_data = builder.build_from_ast(ast_data)
        
        self.nodes = cfg_data['nodes']
        self.edges = cfg_data['edges']
        self.basic_blocks = cfg_data.get('basic_blocks', {})
        self.entry_node_id = cfg_data.get('entry_id')
        self.exit_node_id = cfg_data.get('exit_id')
        
        return self

    def get_execution_paths(self, max_paths: int = 50) -> List[List[int]]:
        """Find all execution paths from entry to exit"""
        if not self.entry_node_id or not self.exit_node_id:
            return []
        
        paths = []
        stack = [(self.entry_node_id, [self.entry_node_id])]
        visited_cycles = set()
        
        while stack and len(paths) < max_paths:
            node_id, path = stack.pop()
            
            if node_id == self.exit_node_id:
                paths.append(path)
                continue
            
            node = self.nodes.get(node_id)
            if not node:
                continue
            
            for next_id in node.out_edges:
                # Prevent infinite loops
                if next_id in path:
                    cycle_key = f"{node_id}-{next_id}"
                    if cycle_key in visited_cycles:
                        continue
                    visited_cycles.add(cycle_key)
                
                stack.append((next_id, path + [next_id]))
        
        return paths

    def get_cyclomatic_complexity(self) -> int:
        """Calculate cyclomatic complexity: E - N + 2"""
        E = len(self.edges)
        N = len(self.nodes)
        return E - N + 2

    def get_control_structures_count(self) -> Dict[str, int]:
        """Count control structures"""
        counts = {
            'decisions': 0,
            'loops': 0,
            'functions': 0,
            'returns': 0
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
        
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'entry_node': self.entry_node_id,
            'exit_node': self.exit_node_id,
            'cyclomatic_complexity': self.get_cyclomatic_complexity(),
            'control_structures': self.get_control_structures_count()
        }


def create_mock_ast() -> Dict:
    """Create mock AST for testing"""
    return {
        'type': 'Module',
        'children': [
            {
                'type': 'FunctionDef',
                'value': 'test_function',
                'line': 1,
                'children': [
                    {
                        'type': 'If',
                        'line': 2,
                        'children': [
                            {'type': 'test', 'value': 'condition'},
                            {
                                'type': 'body',
                                'children': [
                                    {'type': 'Assign', 'line': 3},
                                    {'type': 'Return', 'line': 4}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }