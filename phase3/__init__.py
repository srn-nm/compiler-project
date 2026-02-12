"""
Phase 3 Package - Control Flow Graph Analysis
"""

# Import از ماژول‌های داخلی
from .analyzer.cfg_analyzer import CFGAnalyzer, Phase3CFGSimilarity
from .visualizer.cfg_visualizer import visualize_cfg, generate_cfg_report
from .integration.phase_integration import Phase3Integration

# Export توابع به صورت مستقیم
__version__ = "1.0.0"

CFGAnalyzer = CFGAnalyzer
Phase3CFGSimilarity = Phase3CFGSimilarity
visualize_cfg = visualize_cfg
generate_cfg_report = generate_cfg_report
Phase3Integration = Phase3Integration

__all__ = [
    'CFGAnalyzer',
    'Phase3CFGSimilarity',
    'visualize_cfg',
    'generate_cfg_report',
    'Phase3Integration'
]