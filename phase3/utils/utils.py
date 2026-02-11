"""
Utility functions for phase 3
"""

import json
import hashlib
import re
import math
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from collections import Counter


def normalize_code(code: str, language: str = 'python') -> str:
    """
    Normalize code for comparison

    Args:
        code: Source code
        language: Programming language

    Returns:
        Normalized code string
    """
    if language == 'python':
        return normalize_python_code(code)
    elif language in ['java', 'cpp', 'c']:
        return normalize_c_like_code(code)
    else:
        return normalize_general_code(code)


def normalize_python_code(code: str) -> str:
    # Remove comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)

    # Normalize whitespace
    code = re.sub(r'\s+', ' ', code)

    # Normalize string literals
    code = re.sub(r'".*?"', '"STRING"', code)
    code = re.sub(r"'.*?'", "'STRING'", code)

    # Normalize numbers
    code = re.sub(r'\b\d+\b', 'NUMBER', code)
    code = re.sub(r'\b\d+\.\d+\b', 'FLOAT', code)

    # Normalize variable names (optional)
    # code = re.sub(r'\b[a-z_][a-z0-9_]*\b', 'VAR', code, flags=re.IGNORECASE)

    return code.strip()


def normalize_c_like_code(code: str) -> str:
    """C-like code (C, C++, Java)"""
    # Remove comments
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    # Normalize whitespace
    code = re.sub(r'\s+', ' ', code)

    # Normalize string literals
    code = re.sub(r'".*?"', '"STRING"', code)

    # Normalize numbers
    code = re.sub(r'\b\d+\b', 'NUMBER', code)
    code = re.sub(r'\b\d+\.\d+\b', 'FLOAT', code)

    return code.strip()


def normalize_general_code(code: str) -> str:
    """Normalize code for any language"""
    # Remove extra whitespace
    code = re.sub(r'\s+', ' ', code)

    # Remove common comment patterns
    comment_patterns = [
        r'#.*$',          # Python, Shell
        r'//.*$',         # C, C++, Java, JavaScript
        r'/\*.*?\*/',     # Multi-line comments
        r'--.*$',         # SQL
        r'<!--.*?-->',    # HTML
    ]

    for pattern in comment_patterns:
        code = re.sub(pattern, '', code, flags=re.MULTILINE | re.DOTALL)

    return code.strip()


def calculate_hash(data: Any) -> str:
    """
    Args:
        data: Any data that can be converted to string

    Returns:
        MD5 hash string
    """
    data_str = str(data).encode('utf-8')
    return hashlib.md5(data_str).hexdigest()


def calculate_similarity_percentage(value1: Any, value2: Any,
                                   metric: str = 'jaccard') -> float:
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


def extract_code_metadata(code: str, language: str = 'python') -> Dict[str, Any]:
    metadata = {
        'length': len(code),
        'lines': code.count('\n') + 1,
        'language': language,
        'hash': calculate_hash(code)
    }

    # Count basic elements
    if language == 'python':
        metadata['functions'] = len(re.findall(r'def\s+\w+\s*\(', code))
        metadata['classes'] = len(re.findall(r'class\s+\w+', code))
        metadata['imports'] = len(re.findall(r'import\s+', code))
        metadata['loops'] = len(re.findall(r'\b(for|while)\b', code))
        metadata['conditionals'] = len(re.findall(r'\b(if|elif)\b', code))
    elif language in ['java', 'cpp', 'c']:
        metadata['functions'] = len(re.findall(r'\w+\s+\w+\s*\([^)]*\)\s*\{', code))
        metadata['classes'] = len(re.findall(r'class\s+\w+', code))
        metadata['loops'] = len(re.findall(r'\b(for|while)\b', code))
        metadata['conditionals'] = len(re.findall(r'\b(if|else\s*if)\b', code))

    return metadata


def save_to_json(data: Any, filename: str, indent: int = 2) -> str:

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

    return filename


def load_from_json(filename: str) -> Any:
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_percentage(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}%"


def format_decision(is_similar: bool, threshold: float = 70.0) -> Tuple[str, str]:
    if is_similar:
        return ("Similar (Possible plagiarism)", "🔴")
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

    # Valid extensions
    valid_extensions = [
        '.py', '.java', '.cpp', '.c', '.cc', '.cxx', '.h', '.hpp',
        '.js', '.ts', '.go', '.rs', '.php', '.rb', '.cs'
    ]

    if path.suffix.lower() not in valid_extensions:
        return False, f"Invalid file extension. Valid: {', '.join(valid_extensions)}"

    if path.stat().st_size > 10 * 1024 * 1024:
        return False, "File is too large (maximum 10 MB)"

    return True, "File is valid"


def extract_keywords(code: str, language: str = 'python') -> List[str]:
    """
    Extract keywords from code

    """
    keyword_patterns = {
        'python': [
            'def', 'class', 'if', 'else', 'elif', 'for', 'while',
            'return', 'import', 'from', 'as', 'try', 'except',
            'finally', 'with', 'assert', 'raise', 'yield', 'lambda',
            'pass', 'break', 'continue', 'global', 'nonlocal'
        ],
        'java': [
            'public', 'private', 'protected', 'class', 'interface',
            'void', 'static', 'final', 'if', 'else', 'for', 'while',
            'return', 'try', 'catch', 'finally', 'throw', 'throws',
            'import', 'package', 'new', 'this', 'super'
        ],
        'cpp': [
            'int', 'float', 'double', 'char', 'void', 'bool',
            'if', 'else', 'for', 'while', 'do', 'switch', 'case',
            'return', 'class', 'struct', 'public', 'private',
            'protected', 'template', 'namespace', 'using'
        ]
    }

    keywords = keyword_patterns.get(language, [])
    found_keywords = []

    for keyword in keywords:
        pattern = r'\b' + keyword + r'\b'
        if re.search(pattern, code):
            found_keywords.append(keyword)

    return found_keywords


def calculate_code_complexity(code: str, language: str = 'python') -> Dict[str, int]:
    """
    Calculate code complexity metrics
    """
    metrics = {
        'lines': code.count('\n') + 1,
        'characters': len(code),
        'words': len(re.findall(r'\b\w+\b', code))
    }

    # Count control structures
    if language == 'python':
        metrics['functions'] = len(re.findall(r'def\s+\w+\s*\(', code))
        metrics['classes'] = len(re.findall(r'class\s+\w+', code))
        metrics['loops'] = len(re.findall(r'\b(for|while)\b', code))
        metrics['conditionals'] = len(re.findall(r'\b(if|elif|else)\b', code))
        metrics['try_blocks'] = len(re.findall(r'\btry\b', code))
    else:
        metrics['functions'] = len(re.findall(r'\w+\s+\w+\s*\([^)]*\)\s*\{', code))
        metrics['loops'] = len(re.findall(r'\b(for|while|do)\b', code))
        metrics['conditionals'] = len(re.findall(r'\b(if|else|switch)\b', code))
        metrics['try_blocks'] = len(re.findall(r'\btry\b', code))

    # Calculate estimated cyclomatic complexity
    metrics['estimated_complexity'] = (
        metrics.get('functions', 0) +
        metrics.get('loops', 0) +
        metrics.get('conditionals', 0) +
        metrics.get('try_blocks', 0) + 1
    )

    return metrics


def merge_results(*results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple result dictionaries

    """
    merged = {}

    for results in results_list:
        for key, value in results.items():
            if key in merged:
                if isinstance(value, dict) and isinstance(merged[key], dict):
                    merged[key].update(value)
                elif isinstance(value, list) and isinstance(merged[key], list):
                    merged[key].extend(value)
                else:
                    merged[key] = value
            else:
                merged[key] = value

    return merged


def create_summary_report(results: Dict[str, Any]) -> str:
    """
    Create a summary report from analysis results

    """
    report_lines = []

    report_lines.append("=" * 70)
    report_lines.append("CODE SIMILARITY ANALYSIS - SUMMARY REPORT")
    report_lines.append("=" * 70)

    if 'combined_similarity_score' in results:
        score = results['combined_similarity_score']
        report_lines.append(f"\nOverall Similarity Score: {score:.2f}%")
    elif 'cfg_similarity_score' in results:
        score = results['cfg_similarity_score']
        report_lines.append(f"\nCFG Similarity Score: {score:.2f}%")

    is_similar = results.get('is_plagiarism_suspected', False)
    decision_text, emoji = format_decision(is_similar)
    report_lines.append(f"Detection Result: {emoji} {decision_text}")

    if 'individual_scores' in results:
        report_lines.append("\nIndividual Phase Scores:")
        scores = results['individual_scores']
        for phase, score in scores.items():
            report_lines.append(f"  • {phase.upper()}: {score:.2f}%")

    if 'detailed_metrics' in results:
        report_lines.append("\nDetailed Metrics:")
        metrics = results['detailed_metrics']
        for metric, value in metrics.items():
            if isinstance(value, float):
                report_lines.append(f"  • {metric}: {value:.3f}")

    if 'similar_components' in results and results['similar_components']:
        components = results['similar_components']
        report_lines.append(f"\nSimilar Components Found: {len(components)}")
        for i, comp in enumerate(components[:5], 1):
            report_lines.append(f"  {i}. {comp.get('type', 'unknown')} "
                              f"(similarity: {comp.get('similarity', 0):.2f})")

    report_lines.append("\n" + "=" * 70)

    return "\n".join(report_lines)