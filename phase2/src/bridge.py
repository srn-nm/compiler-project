#Bridge between Phase 1 (Token) and Phase 2 (AST)

import json
from typing import Dict, Optional, Any
from .analyzer import Phase2ASTSimilarity

class Phase1Phase2Bridge:
    
    def __init__(self, phase1_results_path: Optional[str] = None):
        self.phase1_results = None
        self.phase2_analyzer = None
        
        if phase1_results_path:
            self.load_phase1_results(phase1_results_path)
    
    def load_phase1_results(self, file_path: str) -> bool:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.phase1_results = json.load(f)
            print(f" Phase 1 results loaded: {self.phase1_results.get('overall_similarity', 0):.1f}% similarity")
            return True
        except Exception as e:
            print(f" Failed to load Phase 1 results: {e}")
            return False
    
    def run_phase2_with_phase1(self, code1: str, code2: str, 
                               language: str = 'python',
                               config_path: Optional[str] = None) -> Dict[str, Any]:
     
        self.phase2_analyzer = Phase2ASTSimilarity(config_path)
        
        results = self.phase2_analyzer.analyze_code_pair(
            code1, code2, language, self.phase1_results
        )
        
        return results
    
    def get_focus_areas(self) -> list:
        if not self.phase1_results:
            return []
        
        focus_areas = []
        matches = self.phase1_results.get('matched_sections', [])
        
        for match in matches[:10]:
            focus_areas.append({
                'file1_lines': (match.get('line_numbers', {}).get('start1'), 
                               match.get('line_numbers', {}).get('end1')),
                'file2_lines': (match.get('line_numbers', {}).get('start2'),
                               match.get('line_numbers', {}).get('end2')),
                'similarity': match.get('similarity', 0),
                'length': match.get('length', 0)
            })
        
        return focus_areas
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.phase1_results:
            return {}
        
        return {
            'token_similarity': self.phase1_results.get('overall_similarity', 0),
            'normalized_similarity': self.phase1_results.get('normalized_similarity', 0),
            'matching_sections': len(self.phase1_results.get('matched_sections', [])),
            'common_variables': self.phase1_results.get('variable_patterns', {}).get('common_count', 0)
        }



def run_integrated_analysis(code1: str, code2: str, phase1_json_path: Optional[str] = None, language: str = 'python') -> Dict[str, Any]:

    bridge = Phase1Phase2Bridge(phase1_json_path)
    return bridge.run_phase2_with_phase1(code1, code2, language)

def analyze_with_focus(code1: str, code2: str, phase1_json_path: str, focus_on_matches: bool = True) -> Dict[str, Any]:
    
    bridge = Phase1Phase2Bridge(phase1_json_path)
    
    if focus_on_matches:
        focus_areas = bridge.get_focus_areas()
        print(f" Focusing on {len(focus_areas)} high-similarity regions")
    
    return bridge.run_phase2_with_phase1(code1, code2)


def quick_bridge(phase1_json: str, file1: str, file2: str) -> Dict[str, Any]:
    with open(file1, 'r', encoding='utf-8') as f:
        code1 = f.read()
    with open(file2, 'r', encoding='utf-8') as f:
        code2 = f.read()
    
    return run_integrated_analysis(code1, code2, phase1_json)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Bridge Phase 1 -> Phase 2')
    parser.add_argument('--phase1', required=True, help='Phase 1 JSON results file')
    parser.add_argument('--file1', required=True, help='First source code file')
    parser.add_argument('--file2', required=True, help='Second source code file')
    parser.add_argument('--output', '-o', default='phase2_results.json', help='Output file')
    
    args = parser.parse_args()
    
    # run bridge
    results = quick_bridge(args.phase1, args.file1, args.file2)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n Bridge completed!")
    print(f"   Token Similarity: {results.get('token_similarity_score', 0):.1f}%")
    print(f"   AST Similarity: {results.get('ast_similarity_score', 0):.1f}%")
    print(f"   Combined: {results.get('combined_similarity_score', 0):.1f}%")
    print(f"   Decision: {results.get('final_decision', 'UNKNOWN')}")
    print(f"\n Results saved to: {args.output}")