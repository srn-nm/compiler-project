"""
Phase 3 integration with phases 1 and 2
"""

import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ..analyzer.cfg_analyzer import Phase3CFGSimilarity


class Phase3Integration:
    """Phase 3 integration class"""

    @staticmethod
    def integrate_all_phases(code1: str, code2: str,
                             phase1_results: Optional[Dict] = None,
                             phase2_results: Optional[Dict] = None,
                             phase3_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Integrate results from all three phases

        Parameters:
            code1: First code
            code2: Second code
            phase1_results: Phase 1 results
            phase2_results: Phase 2 results
            phase3_results: Phase 3 results

        """
        if phase3_results is None:
            phase3 = Phase3CFGSimilarity()
            phase3_results = phase3.analyze_code_pair(code1, code2, phase1_results, phase2_results)

        if phase1_results and phase2_results and phase3_results:
            return Phase3Integration._combine_all_results(
                phase1_results, phase2_results, phase3_results
            )

        return phase3_results

    @staticmethod
    def _combine_all_results(phase1: Dict, phase2: Dict, phase3: Dict) -> Dict[str, Any]:
        """Combine results from all three phases"""
        token_score = Phase3Integration._extract_score(phase1, 'overall_similarity')
        ast_score = Phase3Integration._extract_score(phase2, 'ast_similarity_score')
        cfg_score = Phase3Integration._extract_score(phase3, 'cfg_similarity_score')

        weights = {
            'token': 0.2,
            'ast': 0.3,
            'cfg': 0.5
        }

        combined_score = (
                                 weights['token'] * token_score +
                                 weights['ast'] * ast_score +
                                 weights['cfg'] * cfg_score
                         ) * 100

        confidence = Phase3Integration._calculate_confidence(
            token_score, ast_score, cfg_score
        )

        is_plagiarism = combined_score >= 70

        return {
            'final_analysis': True,
            'combined_similarity_score': combined_score,
            'individual_scores': {
                'token': token_score * 100,
                'ast': ast_score * 100,
                'cfg': cfg_score * 100
            },
            'weights_applied': weights,
            'confidence': confidence,
            'final_decision': 'PLAGIARISM_SUSPECTED' if is_plagiarism else 'CLEAN',
            'is_plagiarism_suspected': is_plagiarism,
            'phase_summaries': {
                'phase1': Phase3Integration._create_phase_summary(phase1, 'token'),
                'phase2': Phase3Integration._create_phase_summary(phase2, 'ast'),
                'phase3': Phase3Integration._create_phase_summary(phase3, 'cfg')
            },
            'recommendation': Phase3Integration._generate_recommendation(
                token_score, ast_score, cfg_score, is_plagiarism
            )
        }

    @staticmethod
    def _extract_score(results: Dict, key: str) -> float:
        """Extract score from results"""
        score = results.get(key, 0)
        if isinstance(score, (int, float)):
            return score / 100 if score > 1 else score
        return 0.0

    @staticmethod
    def _calculate_confidence(token_score: float,
                              ast_score: float,
                              cfg_score: float) -> float:
        """Calculate system confidence level"""
        scores = [token_score, ast_score, cfg_score]

        mean = sum(scores) / 3
        variance = sum((s - mean) ** 2 for s in scores) / 3

        confidence = 1 - min(variance * 5, 1.0)

        confidence *= mean

        return confidence * 100

    @staticmethod
    def _create_phase_summary(results: Dict, phase_type: str) -> Dict[str, Any]:
        """Create summary for each phase"""
        if phase_type == 'token':
            return {
                'type': 'token_analysis',
                'score': results.get('overall_similarity', 0),
                'matched_tokens': results.get('token_counts', {}).get('common', 0),
                'key_findings': results.get('matched_sections', [])[:3]
            }
        elif phase_type == 'ast':
            return {
                'type': 'ast_analysis',
                'score': results.get('ast_similarity_score', 0),
                'node_count': results.get('ast_statistics', {}).get('code1', {}).get('total_nodes', 0),
                'matched_nodes': results.get('matched_nodes_count', 0)
            }
        elif phase_type == 'cfg':
            return {
                'type': 'cfg_analysis',
                'score': results.get('cfg_similarity_score', 0),
                'node_count': results.get('cfg_statistics', {}).get('code1', {}).get('node_count', 0),
                'control_structures': results.get('cfg_statistics', {}).get('code1', {}).get('control_structures', {})
            }
        else:
            return {'type': 'unknown'}

    @staticmethod
    def _generate_recommendation(token_score: float,
                                 ast_score: float,
                                 cfg_score: float,
                                 is_plagiarism: bool) -> str:
        """Generate recommendation based on results"""
        if is_plagiarism:
            if cfg_score > 0.8:
                return "Plagiarism detected with high confidence. Immediate review required."
            elif ast_score > 0.7:
                return "Similar structures identified. Further detailed review recommended."
            else:
                return "Superficial similarity identified. Might be simple inspiration."
        else:
            if cfg_score < 0.3:
                return "Codes are completely different behaviorally."
            elif ast_score < 0.5:
                return "Have different structures. Plagiarism probability is very low."
            else:
                return "Although there are similarities, they are not sufficient for plagiarism detection."

    @staticmethod
    def load_results_from_files(phase1_file: str,
                                phase2_file: str,
                                phase3_file: str = None) -> tuple:
        """Load results from files"""
        results = []

        for file_path in [phase1_file, phase2_file, phase3_file]:
            if file_path and Path(file_path).exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        results.append(json.load(f))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    results.append(None)
            else:
                results.append(None)

        return tuple(results)


def run_complete_analysis(code1: str, code2: str,
                          output_file: str = 'complete_analysis.json') -> Dict[str, Any]:
    """
    This function tries to run all three phases and combine the results

    """
    print("Starting complete project analysis...")
    print("=" * 60)

    phase1_results = None
    phase2_results = None
    phase3_results = None

    # Try to run phase 1
    try:
        print("\n Running phase 1 (token analysis)...")
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))

        from phase1.token_similarity_analyzer import TokenSimilarityAnalyzer
        phase1_analyzer = TokenSimilarityAnalyzer()
        phase1_results = phase1_analyzer.calculate_similarity(code1, code2)
        print(f"Phase 1 completed - Score: {phase1_results.get('overall_similarity', 0):.1f}%")
    except ImportError as e:
        print(f"Phase 1 not found: {e}")
    except Exception as e:
        print(f"Error running phase 1: {e}")

    try:
        print("\nRunning phase 2 (AST analysis)...")
        project_root = Path(__file__).parent.parent.parent
        phase2_path = project_root / 'phase2'
        sys.path.insert(0, str(phase2_path))

        from analyzer import ASTSimilarityAnalyzer
        phase2_analyzer = ASTSimilarityAnalyzer()
        ast1 = phase2_analyzer.parse_code(code1)
        ast2 = phase2_analyzer.parse_code(code2)
        phase2_results = phase2_analyzer.calculate_ast_similarity(ast1, ast2)
        print(f"Phase 2 completed - Score: {phase2_results.get('ast_similarity_score', 0):.1f}%")
    except ImportError as e:
        print(f"Phase 2 not found: {e}")
    except Exception as e:
        print(f"Error running phase 2: {e}")

    print("\nRunning phase 3 (CFG analysis)...")
    phase3 = Phase3CFGSimilarity()
    phase3_results = phase3.analyze_code_pair(code1, code2, phase1_results, phase2_results)
    print(f"Phase 3 completed - Score: {phase3_results.get('cfg_similarity_score', 0):.1f}%")

    print("\n Combining results from all three phases...")
    final_results = Phase3Integration.integrate_all_phases(
        code1, code2, phase1_results, phase2_results, phase3_results
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\nComplete results saved in {output_file}.")

    return final_results