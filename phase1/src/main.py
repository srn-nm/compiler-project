import json
from token_similarity_analyzer import TokenSimilarityAnalyzer
import argparse

def read_code_from_file(filename):
    """Read code from file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser(description='Token-based Code Similarity Analyzer')
    parser.add_argument('file1', help='First Python file')
    parser.add_argument('file2', help='Second Python file')
    parser.add_argument('--output', '-o', help='Output JSON file', default='similarity_report.json')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    code1 = read_code_from_file(args.file1)
    code2 = read_code_from_file(args.file2)
    
    # Analyze similarity
    analyzer = TokenSimilarityAnalyzer()
    result = analyzer.calculate_similarity(code1, code2)
    
    # Output results
    if args.verbose:
        print("=" * 60)
        print("CODE SIMILARITY ANALYSIS REPORT")
        print("=" * 60)
        print(f"\nOverall Similarity: {result['overall_similarity']:.2f}%")
        print("\nDetailed Metrics:")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nToken Counts:")
        print(f"  File 1: {result['token_counts']['code1']} tokens")
        print(f"  File 2: {result['token_counts']['code2']} tokens")
        print(f"  Common token types: {result['token_counts']['common']}")
        
        if result['matched_sections']:
            print(f"\nMatched Sections (≥5 consecutive tokens):")
            for i, match in enumerate(result['matched_sections'][:5], 1):
                print(f"  Match {i}: {match['length']} tokens")
                print(f"    File1[{match['start1']}:{match['end1']}] = File2[{match['start2']}:{match['end2']}]")
                if args.verbose:
                    print(f"    Tokens: {' '.join(match['tokens'][:3])}...")
    
    # Save to JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\nReport saved to {args.output}")

if __name__ == "__main__":
    main()