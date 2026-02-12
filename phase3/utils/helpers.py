"""
Utility functions for Phase 3
Lightweight and focused on CFG analysis
"""

import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime


def normalize_code(code: str, language: str = 'python') -> str:
    """Simple code normalization"""
    # Remove comments
    lines = []
    for line in code.split('\n'):
        if '#' in line:
            line = line[:line.index('#')]
        if line.strip():
            lines.append(line.strip())
    
    return '\n'.join(lines)


def calculate_similarity(val1: float, val2: float) -> float:
    """Calculate similarity between two values"""
    if val1 == 0 and val2 == 0:
        return 1.0
    max_val = max(val1, val2)
    return 1 - abs(val1 - val2) / max_val if max_val > 0 else 0.0


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format as percentage string"""
    return f"{value:.{decimals}f}%"


def format_decision(is_similar: bool, threshold: float = 70) -> Tuple[str, str]:
    """Format decision with emoji"""
    if is_similar:
        if threshold >= 80:
            return ("High confidence plagiarism", "🔴")
        elif threshold >= 70:
            return ("Possible plagiarism", "🟠")
        else:
            return ("Suspicious", "🟡")
    else:
        return ("Original work", "🟢")


def save_json(data: Any, filename: str, indent: int = 2) -> str:
    """Save data to JSON file"""
    if not filename.endswith('.json'):
        filename += '.json'
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    
    return filename


def load_json(filename: str) -> Any:
    """Load data from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_summary(results: Dict[str, Any]) -> str:
    """Create human-readable summary"""
    lines = []
    lines.append("=" * 70)
    lines.append("CODE SIMILARITY ANALYSIS - SUMMARY")
    lines.append("=" * 70)
    
    # Score
    if 'combined_similarity_score' in results:
        score = results['combined_similarity_score']
        lines.append(f"\n🎯 Overall Similarity: {score:.2f}%")
    elif 'cfg_similarity_score' in results:
        score = results['cfg_similarity_score']
        lines.append(f"\n🎯 CFG Similarity: {score:.2f}%")
    
    # Decision
    is_similar = results.get('is_plagiarism_suspected', False)
    threshold = results.get('threshold_used', 70)
    decision, emoji = format_decision(is_similar, threshold)
    lines.append(f"\n⚖️  Verdict: {emoji} {decision}")
    
    # Timestamp
    lines.append(f"\n📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)