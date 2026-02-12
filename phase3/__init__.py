"""
Phase 3 Package - Control Flow Graph Analysis
Complete plagiarism detection system
"""

from .analyzer.cfg_analyzer import CFGAnalyzer, Phase3CFGSimilarity
from phase3.analyzer.graph_similarity import GraphSimilarity
from .integration.phase_integration import Phase3Integration
from .visualizer.cfg_visualizer import visualize_cfg, generate_cfg_report, generate_cfg_dot_file
from .utils.helpers import (
    normalize_code, calculate_similarity, 
    format_percentage, format_decision,
    save_json, load_json, create_summary
)

__version__ = "3.0.0"
__all__ = [
    'CFGAnalyzer',
    'Phase3CFGSimilarity',
    'GraphSimilarity',
    'Phase3Integration',
    'visualize_cfg',
    'generate_cfg_report',
    'generate_cfg_dot_file',
    'normalize_code',
    'calculate_similarity',
    'format_percentage',
    'format_decision',
    'save_json',
    'load_json',
    'create_summary'
]