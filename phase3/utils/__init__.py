"""
Utility functions package for Phase 3
"""

from .utils import (
    normalize_code,
    normalize_python_code,
    normalize_c_like_code,
    normalize_general_code,
    calculate_hash,
    calculate_similarity_percentage,
    levenshtein_similarity,
    extract_code_metadata,
    save_to_json,
    load_from_json,
    format_percentage,
    format_decision,
    validate_code_file,
    extract_keywords,
    calculate_code_complexity,
    merge_results,
    create_summary_report
)

__version__ = "1.0.0"
__all__ = [
    'normalize_code',
    'normalize_python_code',
    'normalize_c_like_code',
    'normalize_general_code',
    'calculate_hash',
    'calculate_similarity_percentage',
    'levenshtein_similarity',
    'extract_code_metadata',
    'save_to_json',
    'load_from_json',
    'format_percentage',
    'format_decision',
    'validate_code_file',
    'extract_keywords',
    'calculate_code_complexity',
    'merge_results',
    'create_summary_report'
]