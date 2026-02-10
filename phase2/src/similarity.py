"""
Similarity calculation and integration functions for Phase 2
"""

import json
from typing import Dict, Any, Optional
from .analyzer import Phase2ASTSimilarity


def calculate_ast_similarity(code1: str, code2: str, language: str = 'python',
                           config_path: str = None) -> Dict[str, Any]:
    """Calculate AST similarity between two codes"""
    analyzer = Phase2ASTSimilarity(config_path)
    return analyzer.analyze_code_pair(code1, code2, language)


def analyze_code_pair(code1: str, code2: str, language: str = 'python',
                     phase1_results: Optional[Dict] = None,
                     config_path: str = None) -> Dict[str, Any]:
    """Complete analysis of a code pair"""
    analyzer = Phase2ASTSimilarity(config_path)
    return analyzer.analyze_code_pair(code1, code2, language, phase1_results)


def integrate_with_phase1(phase1_results: Dict, code1: str, code2: str,
                         language: str = 'python', config_path: str = None) -> Dict[str, Any]:
    """
    Combine Phase 1 and Phase 2 results
    """
    analyzer = Phase2ASTSimilarity(config_path)
    return analyzer.analyze_code_pair(code1, code2, language, phase1_results)


def run_phase1_and_phase2(code1: str, code2: str, language: str = 'python',
                         phase1_config: str = None, phase2_config: str = None) -> Dict[str, Any]:
    """
    Automatically run both phases
    This function attempts to run Phase 1 and then combine results with Phase 2.
    """
    phase1_results = None

    try:
        import sys
        import os

        phase1_path = os.path.join(os.path.dirname(__file__), '..', 'phase1')
        if os.path.exists(phase1_path):
            sys.path.insert(0, phase1_path)

            try:
                from src.token_similarity_analyzer import TokenSimilarityAnalyzer

                token_analyzer = TokenSimilarityAnalyzer()

                if phase1_config and os.path.exists(phase1_config):
                    with open(phase1_config, 'r') as f:
                        phase1_config_data = json.load(f)
                    # TODO: Apply config to analyzer

                phase1_results = token_analyzer.calculate_similarity(code1, code2)

                print("Phase 1 executed successfully")

            except ImportError as e:
                print(f"Could not import Phase 1: {e}")
                print("Only Phase 2 will be executed")

    except Exception as e:
        print(f"Error in executing Phase 1: {e}")

    return analyze_code_pair(code1, code2, language, phase1_results, phase2_config)


def save_results(results: Dict[str, Any], output_file: str = 'phase2_results.json'):
    """Save results to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    txt_file = output_file.replace('.json', '.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(generate_text_report(results))

    return output_file, txt_file


def generate_text_report(results: Dict[str, Any]) -> str:
    """Generate text report from results"""
    lines = []
    lines.append("=" * 70)
    lines.append("Phase 2 Analysis Report - Structural Similarity (AST)")
    lines.append("=" * 70)

    if 'phase_integration' in results:
        lines.append(f"Analysis Type: {results['phase_integration']}")
        lines.append(f"Combined Score: {results.get('combined_similarity_score', 0):.2f}%")
        lines.append(f"  • Token Score: {results.get('token_similarity_score', 0):.2f}%")
        lines.append(f"  • Structural Score: {results.get('ast_similarity_score', 0):.2f}%")
    else:
        lines.append(f"Structural (AST) Score: {results.get('ast_similarity_score', 0):.2f}%")

    lines.append(f"Detection Threshold: {results.get('threshold', 0.65) * 100:.0f}%")

    is_suspected = results.get('is_plagiarism_suspected', False)
    decision = results.get('final_decision', 'UNKNOWN')
    lines.append(f"Detection: {'Similar (Possible Plagiarism)' if is_suspected else '🟢 Not Similar'}")
    if decision != 'UNKNOWN':
        lines.append(f"Final Decision: {decision}")

    if 'phase2_details' in results:
        phase2 = results['phase2_details']
        lines.append("\nStructural Analysis Details:")

        if 'ast_statistics' in phase2:
            stats1 = phase2['ast_statistics'].get('code1', {})
            stats2 = phase2['ast_statistics'].get('code2', {})
            lines.append(f"  • Code 1 Node Count: {stats1.get('total_nodes', 0)}")
            lines.append(f"  • Code 2 Node Count: {stats2.get('total_nodes', 0)}")
            lines.append(f"  • Similar Nodes: {phase2.get('matched_nodes_count', 0)}")

    if 'phase2_details' in results and 'ast_similarity_metrics' in results['phase2_details']:
        metrics = results['phase2_details']['ast_similarity_metrics']
        lines.append("\nDetailed Metrics:")
        for key, value in metrics.items():
            lines.append(f"  • {key}: {value:.3f}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)