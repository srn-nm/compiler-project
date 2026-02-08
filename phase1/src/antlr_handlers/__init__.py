import json
import argparse
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from token_similarity_analyzer import TokenSimilarityAnalyzer

def read_code_from_file(filename: str) -> str:
    """Read code from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # If utf-8 fails, try latin-1
        with open(filename, 'r', encoding='latin-1') as f:
            return f.read()


def detect_language_from_filename(filename: str) -> str:
    """Detect language from file extension"""
    ext = Path(filename).suffix.lower()
    
    extensions_map = {
        '.py': 'python',
        '.java': 'java',
        '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', '.hpp': 'cpp', '.h': 'cpp',
        '.c': 'c',
        '.js': 'javascript', '.ts': 'typescript',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.cs': 'csharp',
    }
    
    return extensions_map.get(ext, 'unknown')


def main():
    parser = argparse.ArgumentParser(
        description='Token-based Code Similarity Analyzer - Phase 1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s code1.py code2.py
  %(prog)s file1.java file2.java --verbose
  %(prog)s *.py --matrix --output report.json
        """
    )
    
    parser.add_argument('files', nargs='+', help='Code file(s) for analysis')
    parser.add_argument('--output', '-o', default='similarity_report.json', help='Output JSON file (default: similarity_report.json)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--matrix', action='store_true', help='Matrix comparison of all files with each other')
    parser.add_argument('--config', '-c', help='JSON configuration file')
    parser.add_argument('--language', '-l', choices=['python', 'java', 'cpp', 'c', 'auto'], default='auto', help='Code language (default: auto)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # Read codes
    codes = []
    languages = []
    
    for file in args.files:
        code = read_code_from_file(file)
        codes.append(code)
        
        # Detect language
        if args.language == 'auto':
            detected_lang = detect_language_from_filename(file)
            languages.append(detected_lang)
        else:
            languages.append(args.language)
    
    # Create analyzer
    analyzer = TokenSimilarityAnalyzer(config)
    
    if args.matrix and len(codes) > 1:
        # Matrix analysis
        print("🔍 Performing matrix analysis...")
        results = analyzer.compare_multiple_codes(codes)
        
        if args.verbose:
            print("\nSimilarity Matrix:")
            matrix = results['similarity_matrix']
            for i in range(len(matrix)):
                row = [f"{val*100:5.1f}%" for val in matrix[i]]
                print(f"  Code {i+1}: {row}")
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {args.output}")
        
    elif len(codes) == 2:
        # Pairwise analysis
        print(f"Analyzing similarity between two files...")
        result = analyzer.calculate_similarity(codes[0], codes[1])
        
        # Display results
        if args.verbose:
            print("=" * 70)
            print("Code Similarity Analysis Report")
            print("=" * 70)
            
            print(f"\nLanguages:")
            print(f"  File 1: {result['languages']['code1']}")
            print(f"  File 2: {result['languages']['code2']}")
            
            if result['languages']['mismatch']:
                print("   Warning: Different languages!")
            
            print(f"\nOverall Similarity: {result['overall_similarity']:.2f}%")
            
            print(f"\nDetailed Metrics:")
            for metric, value in result['similarity_metrics'].items():
                print(f"  {metric:15}: {value:.4f}")
            
            print(f"\nToken Statistics:")
            counts = result['token_counts']
            print(f"  File 1: {counts['code1']} tokens")
            print(f"  File 2: {counts['code2']} tokens")
            print(f"  Common types: {counts['common_types']}")
            
            if result['matched_sections']:
                print(f"\nSimilar Sections:")
                for i, match in enumerate(result['matched_sections'][:3], 1):
                    print(f"  Match {i}: {match['length']} tokens")
                    tokens_sample = ' '.join(match['tokens'][:3])
                    if len(match['tokens']) > 3:
                        tokens_sample += ' ...'
                    print(f"    Sample: {tokens_sample}")
            
            if result['common_functions']:
                print(f"\nSimilar Functions:")
                for func in result['common_functions']:
                    print(f"  {func['name']}: {func['similarity']*100:.1f}% similarity")
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {args.output}")
        
    else:
        print("Error: For pairwise analysis exactly 2 files, or for matrix analysis at least 2 files are required.")
        sys.exit(1)


if __name__ == "__main__":
    main()