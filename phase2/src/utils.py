import re
import ast
import json
import hashlib
import math
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
from collections import Counter


def preprocess_code(code: str, language: str = 'python',
                   normalize_identifiers: bool = True,
                   normalize_literals: bool = True,
                   remove_comments: bool = True) -> str:
    """
    Preprocess code before analysis

    """
    if language == 'python':
        return preprocess_python_code(code, normalize_identifiers,
                                     normalize_literals, remove_comments)
    elif language == 'java':
        return preprocess_java_code(code, normalize_identifiers,
                                   normalize_literals, remove_comments)
    elif language in ['cpp', 'c']:
        return preprocess_cpp_code(code, normalize_identifiers,
                                 normalize_literals, remove_comments)
    else:
        # General preprocessing
        return preprocess_general_code(code, remove_comments)


def preprocess_python_code(code: str, normalize_identifiers: bool = True,
                          normalize_literals: bool = True,
                          remove_comments: bool = True) -> str:
    """
    Preprocess Python code
    """
    if remove_comments:
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)

    if normalize_identifiers:
        pass

    if normalize_literals:
        pass

    code = re.sub(r'\s+', ' ', code)

    lines = [line.strip() for line in code.split('\n') if line.strip()]

    return '\n'.join(lines)


def preprocess_java_code(code: str, normalize_identifiers: bool = True,
                        normalize_literals: bool = True,
                        remove_comments: bool = True) -> str:
    """
    Preprocess Java code
    """
    if remove_comments:
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    code = re.sub(r'\s+', ' ', code)

    return code.strip()


def preprocess_cpp_code(code: str, normalize_identifiers: bool = True,
                       normalize_literals: bool = True,
                       remove_comments: bool = True) -> str:
    """
    Preprocess C++/C code
    """
    if remove_comments:
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'///.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'\s+', ' ', code)

    return code.strip()


def preprocess_general_code(code: str, remove_comments: bool = True) -> str:
    """
    General preprocessing for other languages
    """
    code = re.sub(r'\s+', ' ', code)
    return code.strip()


def extract_code_blocks(code: str, language: str = 'python') -> List[Dict[str, Any]]:
    """
    Extract code blocks for detailed analysis

    """
    if language == 'python':
        return extract_python_blocks(code)
    elif language == 'java':
        return extract_java_blocks(code)
    elif language in ['cpp', 'c']:
        return extract_cpp_blocks(code)
    else:
        return []


def extract_python_blocks(code: str) -> List[Dict[str, Any]]:
    """
    Extract Python code blocks
    """
    blocks = []

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                block = {
                    'type': 'function',
                    'name': node.name,
                    'start_line': node.lineno,
                    'end_line': get_node_end_line(node),
                    'args': [arg.arg for arg in node.args.args],
                    'body_preview': ast.unparse(node.body[:2]) if hasattr(ast, 'unparse') else str(node.body[:2]),
                    'hash': calculate_block_hash(node)
                }
                blocks.append(block)

            elif isinstance(node, ast.ClassDef):
                block = {
                    'type': 'class',
                    'name': node.name,
                    'start_line': node.lineno,
                    'end_line': get_node_end_line(node),
                    'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                    'hash': calculate_block_hash(node)
                }
                blocks.append(block)

            elif isinstance(node, ast.If) and node.lineno:
                block = {
                    'type': 'if_block',
                    'name': f'if_block_{node.lineno}',
                    'start_line': node.lineno,
                    'end_line': get_node_end_line(node),
                    'hash': calculate_block_hash(node)
                }
                blocks.append(block)

            elif isinstance(node, ast.For) and node.lineno:
                block = {
                    'type': 'for_loop',
                    'name': f'for_loop_{node.lineno}',
                    'start_line': node.lineno,
                    'end_line': get_node_end_line(node),
                    'hash': calculate_block_hash(node)
                }
                blocks.append(block)

            elif isinstance(node, ast.While) and node.lineno:
                block = {
                    'type': 'while_loop',
                    'name': f'while_loop_{node.lineno}',
                    'start_line': node.lineno,
                    'end_line': get_node_end_line(node),
                    'hash': calculate_block_hash(node)
                }
                blocks.append(block)

    except SyntaxError:
        blocks = extract_blocks_with_regex(code, 'python')

    if not blocks:
        blocks.append({
            'type': 'main',
            'name': 'main_block',
            'content': code[:500],
            'hash': hashlib.md5(code.encode()).hexdigest()[:16]
        })

    return blocks


def extract_java_blocks(code: str) -> List[Dict[str, Any]]:
    """
    Extract Java code blocks using regex
    """
    blocks = []

    class_pattern = r'(?:public|private|protected)?\s*(?:abstract\s+)?\s*(?:class|interface|enum)\s+(\w+)[^{]*\{'
    for match in re.finditer(class_pattern, code, re.DOTALL):
        class_name = match.group(1)
        blocks.append({
            'type': 'class',
            'name': class_name,
            'start_pos': match.start(),
            'hash': hashlib.md5(match.group(0).encode()).hexdigest()[:16]
        })

    method_pattern = r'(?:public|private|protected)?\s*(?:static\s+)?\s*(?:\w+\s+)*\s*(\w+)\s*\([^)]*\)\s*(?:throws\s+[^{]*)?\{'
    for match in re.finditer(method_pattern, code, re.DOTALL):
        method_name = match.group(1)
        blocks.append({
            'type': 'method',
            'name': method_name,
            'start_pos': match.start(),
            'hash': hashlib.md5(match.group(0).encode()).hexdigest()[:16]
        })

    return blocks


def extract_cpp_blocks(code: str) -> List[Dict[str, Any]]:
    """
    Extract C++ code blocks using regex
    """
    blocks = []

    class_pattern = r'(?:class|struct)\s+(\w+)[^{]*\{'
    for match in re.finditer(class_pattern, code, re.DOTALL):
        class_name = match.group(1)
        blocks.append({
            'type': 'class',
            'name': class_name,
            'start_pos': match.start(),
            'hash': hashlib.md5(match.group(0).encode()).hexdigest()[:16]
        })

    func_pattern = r'(?:\w+\s+)*\s*(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{'
    for match in re.finditer(func_pattern, code, re.DOTALL):
        func_name = match.group(1)
        blocks.append({
            'type': 'function',
            'name': func_name,
            'start_pos': match.start(),
            'hash': hashlib.md5(match.group(0).encode()).hexdigest()[:16]
        })

    return blocks


def extract_blocks_with_regex(code: str, language: str = 'python') -> List[Dict[str, Any]]:
    """
    Extract blocks using regex (alternative method)
    """
    blocks = []

    if language == 'python':
        func_pattern = r'def\s+(\w+)\s*\([^)]*\)\s*:'
        for match in re.finditer(func_pattern, code):
            func_name = match.group(1)
            func_body = extract_python_function_body_regex(code, match.start())

            blocks.append({
                'type': 'function',
                'name': func_name,
                'content': func_body,
                'start_pos': match.start(),
                'hash': hashlib.md5(func_body.encode()).hexdigest()[:16]
            })

        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, code):
            class_name = match.group(1)
            blocks.append({
                'type': 'class',
                'name': class_name,
                'start_pos': match.start(),
                'hash': hashlib.md5(class_name.encode()).hexdigest()[:16]
            })

    return blocks


def extract_python_function_body_regex(code: str, start_pos: int) -> str:
    """
    Extract Python function body using regex
    """
    # Find body start
    colon_pos = code.find(':', start_pos)
    if colon_pos == -1:
        return ""

    line_start = code.rfind('\n', 0, colon_pos) + 1
    base_indent = 0
    for i in range(line_start, colon_pos):
        if code[i] == ' ':
            base_indent += 1
        elif code[i] == '\t':
            base_indent += 4  # Assumption: each tab = 4 spaces
        else:
            break

    pos = colon_pos + 1
    while pos < len(code):
        next_newline = code.find('\n', pos)
        if next_newline == -1:
            return code[colon_pos + 1:]

        indent = 0
        line_start = next_newline + 1
        i = line_start
        while i < len(code) and (code[i] == ' ' or code[i] == '\t'):
            if code[i] == ' ':
                indent += 1
            elif code[i] == '\t':
                indent += 4
            i += 1

        if i < len(code) and indent <= base_indent and code[i] != '\n':
            return code[colon_pos + 1:next_newline + 1]

        pos = next_newline + 1

    return code[colon_pos + 1:]


def get_node_end_line(node: ast.AST) -> int:
    """
    Find the end line number of an AST node
    """
    if hasattr(node, 'end_lineno'):
        return node.end_lineno

    last_line = node.lineno

    for child in ast.iter_child_nodes(node):
        if hasattr(child, 'lineno'):
            child_last_line = get_node_end_line(child)
            last_line = max(last_line, child_last_line)

    return last_line


def calculate_block_hash(node: ast.AST) -> str:
    """
    Calculate hash for a code block (structural, without names)
    """
    if hasattr(ast, 'unparse'):
        code = ast.unparse(node)
    else:
        code = ast.dump(node)

    # Normalization: remove variable and function names
    normalized = re.sub(r'\b\w+\b', 'VAR', code)
    normalized = re.sub(r'\d+', 'NUM', normalized)
    normalized = re.sub(r'".*?"', 'STR', normalized)
    normalized = re.sub(r"'.*?'", 'STR', normalized)

    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def calculate_similarity_percentage(value1: Any, value2: Any,
                                   metric: str = 'jaccard') -> float:
    """
    Calculate similarity percentage between two values

    """
    if metric == 'jaccard':
        if isinstance(value1, (list, set)) and isinstance(value2, (list, set)):
            set1 = set(value1) if isinstance(value1, list) else value1
            set2 = set(value2) if isinstance(value2, list) else value2

            if not set1 and not set2:
                return 100.0

            intersection = len(set1 & set2)
            union = len(set1 | set2)

            return (intersection / union * 100) if union > 0 else 0.0
        else:
            return 100.0 if value1 == value2 else 0.0

    elif metric == 'cosine':
        if isinstance(value1, (list, tuple)) and isinstance(value2, (list, tuple)):
            vec1 = Counter(value1)
            vec2 = Counter(value2)

            all_items = set(list(vec1.keys()) + list(vec2.keys()))
            v1 = [vec1.get(item, 0) for item in all_items]
            v2 = [vec2.get(item, 0) for item in all_items]

            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return (dot_product / (norm1 * norm2)) * 100

    elif metric == 'levenshtein':
        if isinstance(value1, str) and isinstance(value2, str):
            return levenshtein_similarity(value1, value2) * 100

    return 0.0


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate Levenshtein similarity between two strings
    """
    if len(s1) < len(s2):
        return levenshtein_similarity(s2, s1)

    if len(s2) == 0:
        return 1.0 if len(s1) == 0 else 0.0

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    distance = previous_row[-1]
    max_len = max(len(s1), len(s2))

    return 1 - (distance / max_len) if max_len > 0 else 1.0


def normalize_identifier(name: str) -> str:
    """
    Normalize identifier names
    """
    name = name.lower()

    patterns = [
        (r'^get_', ''),
        (r'^set_', ''),
        (r'^is_', ''),
        (r'^has_', ''),
        (r'_get$', ''),
        (r'_set$', ''),
        (r'^m_', ''),
        (r'^_', ''),
        (r'_$', ''),
    ]

    for pattern, replacement in patterns:
        name = re.sub(pattern, replacement, name)

    return name


def extract_tokens_from_code(code: str, language: str = 'python') -> List[str]:
    """
    Extract tokens from code (help for Phase 1 integration)
    """
    if language == 'python':
        # Python keywords
        keywords = [
            'def', 'class', 'if', 'else', 'elif', 'for', 'while',
            'return', 'import', 'from', 'as', 'try', 'except',
            'finally', 'with', 'assert', 'raise', 'yield', 'lambda'
        ]

        tokens = []
        words = re.findall(r'\b\w+\b', code)
        for word in words:
            if word in keywords:
                tokens.append(f'KW_{word}')
            elif re.match(r'^[A-Z]', word):
                tokens.append('CLASS_NAME')
            elif re.match(r'^[a-z_][a-z0-9_]*$', word):
                tokens.append('IDENTIFIER')
            elif re.match(r'^\d+$', word):
                tokens.append('NUMBER')

        return tokens

    return []


def save_results_to_file(results: Dict[str, Any], filename: str = 'results.json'):
    """
    Save results to file
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    return filename


def load_results_from_file(filename: str) -> Dict[str, Any]:
    """
    Load results from file
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_percentage(value: float) -> str:
    """
    Format percentage
    """
    return f"{value:.2f}%"


def format_decision(is_suspected: bool) -> Tuple[str, str]:
    """
    Format decision
    """
    if is_suspected:
        return ("Similar", "🔴")
    else:
        return ("Not Similar", "🟢")


def validate_code_file(filepath: str) -> Tuple[bool, str]:
    """
    Validate code file
    """
    path = Path(filepath)

    if not path.exists():
        return False, "File does not exist"

    if not path.is_file():
        return False, "Path is not a file"

    valid_extensions = ['.py', '.java', '.cpp', '.c', '.js', '.go', '.rs', '.php', '.rb', '.cs']
    if path.suffix.lower() not in valid_extensions:
        return False, f"Invalid file extension. Valid extensions: {', '.join(valid_extensions)}"

    if path.stat().st_size > 10 * 1024 * 1024:
        return False, "File is too large (maximum 10 MB)"

    return True, "File is valid"