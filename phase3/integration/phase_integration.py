"""
Phase 3 integration with Phase 1 and Phase 2
Complete three-phase plagiarism detection system
"""

import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase3.analyzer.cfg_analyzer import Phase3CFGSimilarity
from phase3.utils.helpers import save_json, load_json, format_percentage


class Phase3Integration:
    """Complete three-phase analysis integration"""
    
    @staticmethod
    def integrate_all_phases(code1: str, code2: str,
                            phase1_results: Optional[Dict] = None,
                            phase2_results: Optional[Dict] = None,
                            phase3_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Integrate results from all three phases
        """
        # Run Phase 3 if not provided
        if phase3_results is None:
            phase3 = Phase3CFGSimilarity()
            phase3_results = phase3.analyze_code_pair(code1, code2, 
                                                      phase1_results, 
                                                      phase2_results)
        
        # If all phases available, combine them
        if phase1_results and phase2_results and phase3_results:
            return Phase3Integration._combine_results(
                phase1_results, phase2_results, phase3_results
            )
        
        return phase3_results

    @staticmethod
    def _combine_results(phase1: Dict, phase2: Dict, phase3: Dict) -> Dict[str, Any]:
        """Combine results from all three phases with weighted scoring"""
        
        # Extract normalized scores (0-1 range)
        token_score = Phase3Integration._normalize_score(
            phase1.get('overall_similarity', 0)
        )
        ast_score = Phase3Integration._normalize_score(
            phase2.get('ast_similarity_score', 0)
        )
        cfg_score = Phase3Integration._normalize_score(
            phase3.get('cfg_similarity_score', 0)
        )
        
        # Weights: Token 20%, AST 30%, CFG 50%
        weights = {
            'token': 0.2,
            'ast': 0.3,
            'cfg': 0.5
        }
        
        # Calculate combined score
        combined_score = (
            weights['token'] * token_score +
            weights['ast'] * ast_score +
            weights['cfg'] * cfg_score
        ) * 100
        
        # Calculate confidence based on agreement between phases
        confidence = Phase3Integration._calculate_confidence(
            token_score, ast_score, cfg_score
        )
        
        # Determine plagiarism
        threshold = 70  # 70%
        is_plagiarism = combined_score >= threshold
        
        # Generate recommendation
        recommendation = Phase3Integration._generate_recommendation(
            token_score, ast_score, cfg_score, is_plagiarism
        )
        
        return {
            'phase_integration': 'three_phase_complete',
            'combined_similarity_score': round(combined_score, 2),
            'individual_scores': {
                'token': round(token_score * 100, 2),
                'ast': round(ast_score * 100, 2),
                'cfg': round(cfg_score * 100, 2)
            },
            'weights_applied': weights,
            'confidence': round(confidence, 2),
            'final_decision': 'PLAGIARISM_SUSPECTED' if is_plagiarism else 'CLEAN',
            'is_plagiarism_suspected': is_plagiarism,
            'threshold_used': threshold,
            'recommendation': recommendation,
            'phase_summaries': {
                'phase1': Phase3Integration._summarize_phase1(phase1),
                'phase2': Phase3Integration._summarize_phase2(phase2),
                'phase3': Phase3Integration._summarize_phase3(phase3)
            }
        }

    @staticmethod
    def _normalize_score(score: float) -> float:
        """Convert score to 0-1 range"""
        if isinstance(score, (int, float)):
            return score / 100 if score > 1 else score
        return 0.0

    @staticmethod
    def _calculate_confidence(t_score: float, a_score: float, c_score: float) -> float:
        """Calculate confidence based on variance"""
        scores = [t_score, a_score, c_score]
        mean = sum(scores) / 3
        variance = sum((s - mean) ** 2 for s in scores) / 3
        
        # Confidence = 1 - normalized_variance * mean
        confidence = (1 - min(variance * 5, 1.0)) * mean * 100
        return max(0, min(100, confidence))

    @staticmethod
    def _generate_recommendation(t_score: float, a_score: float, 
                                c_score: float, is_plagiarism: bool) -> str:
        """Generate human-readable recommendation"""
        if is_plagiarism:
            if c_score > 0.8:
                return "High-confidence plagiarism detected. Immediate review required."
            elif a_score > 0.7:
                return "Strong structural similarity. Detailed manual review recommended."
            elif t_score > 0.8:
                return "Superficial similarity detected. May be simple code inspiration."
            else:
                return "Plagiarism suspected based on combined evidence."
        else:
            if c_score < 0.3:
                return "Codes are behaviorally different. No plagiarism."
            elif a_score < 0.5:
                return "Different structures and behavior. Likely original work."
            else:
                return "Some similarities exist but insufficient for plagiarism detection."

    @staticmethod
    def _summarize_phase1(results: Dict) -> Dict[str, Any]:
        """Create summary for Phase 1"""
        return {
            'type': 'token_analysis',
            'score': results.get('overall_similarity', 0),
            'matched_tokens': results.get('token_counts', {}).get('common_types', 0),
            'matched_sections': len(results.get('matched_sections', []))
        }

    @staticmethod
    def _summarize_phase2(results: Dict) -> Dict[str, Any]:
        """Create summary for Phase 2"""
        return {
            'type': 'ast_analysis',
            'score': results.get('ast_similarity_score', 0),
            'node_count': results.get('ast_statistics', {})
                          .get('code1', {}).get('total_nodes', 0),
            'matched_nodes': results.get('matched_nodes_count', 0)
        }

    @staticmethod
    def _summarize_phase3(results: Dict) -> Dict[str, Any]:
        """Create summary for Phase 3"""
        return {
            'type': 'cfg_analysis',
            'score': results.get('cfg_similarity_score', 0),
            'node_count': results.get('cfg_statistics', {})
                          .get('code1', {}).get('node_count', 0),
            'matched_paths': results.get('matched_paths_count', 0),
            'similar_components': len(results.get('similar_components', []))
        }


def run_complete_analysis(code1: str, code2: str, 
                         output_file: str = 'complete_analysis.json') -> Dict[str, Any]:
    """
    Run complete three-phase analysis automatically
    """
    print("\n" + "=" * 70)
    print("           COMPLETE THREE-PHASE ANALYSIS")
    print("=" * 70)
    
    phase1_results = None
    phase2_results = None
    
    # ============ Phase 1: Token Analysis ============
    print("\n📊 Phase 1: Token-based analysis...")
    try:
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from phase1.token_similarity_analyzer import TokenSimilarityAnalyzer
        
        analyzer1 = TokenSimilarityAnalyzer()
        phase1_results = analyzer1.calculate_similarity(code1, code2)
        score1 = phase1_results.get('overall_similarity', 0)
        print(f"   ✅ Complete - Score: {score1:.1f}%")
    except ImportError:
        print("   ⚠️ Phase 1 not available - skipping")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    
    # ============ Phase 2: AST Analysis ============
    print("\n🌳 Phase 2: AST analysis...")
    try:
        from phase2.analyzer import ASTSimilarityAnalyzer
        
        analyzer2 = ASTSimilarityAnalyzer()
        ast1 = analyzer2.parse_code(code1)
        ast2 = analyzer2.parse_code(code2)
        phase2_results = analyzer2.calculate_ast_similarity(ast1, ast2)
        score2 = phase2_results.get('ast_similarity_score', 0)
        print(f"   ✅ Complete - Score: {score2:.1f}%")
    except ImportError:
        print("   ⚠️ Phase 2 not available - skipping")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    
    # ============ Phase 3: CFG Analysis ============
    print("\n🔀 Phase 3: CFG analysis...")
    analyzer3 = Phase3CFGSimilarity()
    phase3_results = analyzer3.analyze_code_pair(
        code1, code2, phase1_results, phase2_results
    )
    score3 = phase3_results.get('cfg_similarity_score', 0)
    print(f"   ✅ Complete - Score: {score3:.1f}%")
    
    # ============ Integration ============
    print("\n🔄 Integrating all phases...")
    final_results = Phase3Integration.integrate_all_phases(
        code1, code2, phase1_results, phase2_results, phase3_results
    )
    
    # Save results
    save_json(final_results, output_file)
    print(f"\n💾 Results saved: {output_file}")
    
    return final_results