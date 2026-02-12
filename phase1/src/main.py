import json
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.token_similarity_analyzer import TokenSimilarityAnalyzer


def read_code_from_file(filename: str) -> str:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='latin-1') as f:
            return f.read()


def detect_language_from_filename(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    
    extensions_map = {
        '.py': 'python',
        '.java': 'java',
        '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', '.hpp': 'cpp', '.h': 'cpp',
        '.c': 'c',
    }
    
    return extensions_map.get(ext, 'unknown')


def save_json_report(result: Dict, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"JSON: {output_path}")


def save_html_report(analyzer: TokenSimilarityAnalyzer, result: Dict, 
                    file1_name: str, file2_name: str, output_path: str):
    html_content = analyzer.generate_html_report(result, file1_name, file2_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML: {output_path}")


def save_text_report(result: Dict, file1_name: str, file2_name: str, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("CODE SIMILARITY ANALYSIS REPORT - PHASE 1\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Files:\n")
        f.write(f"  File 1: {file1_name}\n")
        f.write(f"  File 2: {file2_name}\n\n")
        
        f.write(f"Overall Similarity: {result['overall_similarity']:.1f}%\n")
        f.write(f"Normalized Similarity: {result.get('normalized_similarity', 0):.1f}%\n\n")
        
        f.write("Detailed Metrics:\n")
        for metric, value in result['metrics'].items():
            if isinstance(value, (int, float)):
                f.write(f"  {metric:25}: {value*100:.2f}%\n")
        
        f.write("\nToken Statistics:\n")
        counts = result['token_counts']
        f.write(f"  Code 1 tokens: {counts['code1']}\n")
        f.write(f"  Code 2 tokens: {counts['code2']}\n")
        f.write(f"  Common types: {counts['common_types']}\n\n")
        
        f.write(f"Matching Sections: {len(result['matched_sections'])}\n")
        for i, match in enumerate(result['matched_sections'][:5], 1):
            f.write(f"  Match {i}: {match['length']} tokens (lines {match['line_numbers']['start1']}-{match['line_numbers']['end1']})\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")
    
    print(f"Text: {output_path}")


def format_metric_value(value):
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    elif isinstance(value, list):
        return f"[{len(value)} items]"
    elif isinstance(value, dict):
        return f"{{{len(value)} keys}}"
    elif isinstance(value, str):
        return f"\"{value[:50]}\""
    elif value is None:
        return "None"
    else:
        return str(type(value).__name__)


def generate_matrix_html(results: Dict, filenames: List[str]) -> str:
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Code Similarity Matrix Analysis</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
            }}
            .matrix {{
                display: inline-block;
                margin: 20px 0;
            }}
            .matrix-row {{
                display: flex;
            }}
            .matrix-cell {{
                width: 80px;
                height: 60px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 2px;
                border-radius: 4px;
                color: white;
                font-weight: bold;
            }}
            .matrix-label {{
                width: 80px;
                padding: 10px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .stats {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Code Similarity Matrix Analysis</h1>
            
            <div class="stats">
                <strong>Files Analyzed:</strong> {results['num_files']}<br>
                <strong>Comparisons:</strong> {len(results['comparisons'])}<br>
            </div>
            
            <h2>Similarity Matrix</h2>
            <div class="matrix">
                <div class="matrix-row">
                    <div class="matrix-label"></div>
    """
    
    for i in range(results['num_files']):
        html += f'<div class="matrix-label">File {i+1}</div>'
    html += '</div>'
    
    matrix = results['similarity_matrix']
    for i in range(results['num_files']):
        html += f'<div class="matrix-row">'
        html += f'<div class="matrix-label">File {i+1}</div>'
        
        for j in range(results['num_files']):
            val = matrix[i][j]
            color = f'hsl({120 * val}, 70%, 45%)'
            html += f'<div class="matrix-cell" style="background-color: {color};">{val*100:.0f}%</div>'
        
        html += '</div>'
    
    html += '</div>'
    
    html += '<h2>Detailed Comparisons</h2>'
    for comp in results['comparisons'][:10]: 
        html += f'''
        <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px;">
            <strong>File {comp['file1']+1} ↔ File {comp['file2']+1}</strong>: {comp['similarity']*100:.1f}% similarity
            <div style="margin-left: 20px; color: #666; font-size: 14px;">
                {comp['details']['token_counts']['code1']} vs {comp['details']['token_counts']['code2']} tokens,
                {len(comp['details']['matched_sections'])} matching sections
            </div>
        </div>
        '''
    
    if len(results['comparisons']) > 10:
        html += f'<p>... and {len(results["comparisons"]) - 10} more comparisons</p>'
    
    html += f'''
            <p style="text-align: center; color: #666; margin-top: 30px;">
                Generated by TokenSimilarityAnalyzer - Phase 1<br>
                Matrix comparison of {results['num_files']} files
            </p>
        </div>
    </body>
    </html>
    '''
    
    return html


def main():
    parser = argparse.ArgumentParser(
        description='Token-based Code Similarity Analyzer - Phase 1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s code1.py code2.py
  %(prog)s file1.java file2.java --verbose
  %(prog)s file1.cpp file2.cpp --format html
  %(prog)s *.py --matrix --output report.json
        """
    )
    
    parser.add_argument('files', nargs='+', help='Code file(s) for analysis')
    parser.add_argument('--output', '-o', default='similarity_report', 
                       help='Output file name (without extension)')
    parser.add_argument('--format', '-f', choices=['json', 'html', 'text', 'all'], 
                       default='all', help='Output format (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--matrix', action='store_true', help='Matrix comparison of all files')
    parser.add_argument('--config', '-c', help='JSON configuration file')
    parser.add_argument('--language', '-l', choices=['python', 'java', 'cpp', 'c', 'auto'], 
                       default='auto', help='Code language (default: auto)')
    parser.add_argument('--visual', action='store_true', help='Generate visual HTML report')
    
    args = parser.parse_args()
    
    codes = []
    languages = []
    filenames = []
    
    for file in args.files:
        code = read_code_from_file(file)
        codes.append(code)
        filenames.append(os.path.basename(file))
        
        if args.language == 'auto':
            detected_lang = detect_language_from_filename(file)
            languages.append(detected_lang)
        else:
            languages.append(args.language)
    
    analyzer = TokenSimilarityAnalyzer()
    
    if args.matrix and len(codes) > 1:
        print("\nPerforming matrix analysis...")
        results = analyzer.compare_multiple_codes(codes)
        
        if args.verbose:
            print("\nSimilarity Matrix:")
            matrix = results['similarity_matrix']
            print("    " + " ".join([f"File{i+1:3}" for i in range(len(matrix))]))
            for i in range(len(matrix)):
                row = [f"{val*100:5.1f}%" for val in matrix[i]]
                print(f"File{i+1}: {row}")
        
        output_base = args.output
        if args.format in ['json', 'all']:
            save_json_report(results, f"phase1/results/matrix/similarity_report_Phase1_matrix.json")
        if args.format in ['html', 'all'] or args.visual:
            html_content = generate_matrix_html(results, filenames)
            with open(f"phase1/results/matrix/similarity_report_Phase1_matrix.html", 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"HTML: phase1/results/matrix/similarity_report_Phase1_matrix.html")
        
        print(f"\nMatrix analysis complete!")
        
    elif len(codes) == 2:
        print(f"\nAnalyzing similarity between two files...")
        result = analyzer.calculate_similarity(codes[0], codes[1])
        
        if args.verbose:
            print("=" * 70)
            print("CODE SIMILARITY ANALYSIS REPORT - PHASE 1")
            print("=" * 70)
            
            print(f"\nFiles:")
            print(f"  File 1: {filenames[0]} ({languages[0]})")
            print(f"  File 2: {filenames[1]} ({languages[1]})")
            if languages[0] != languages[1]:
                print("Warning: Different languages!")
            
            overall = result['overall_similarity']
            normalized = result.get('normalized_similarity', 0)
            print(f"\nOverall Similarity: {overall:.1f}%")
            print(f"   Normalized Similarity: {normalized:.1f}% (ignoring variable names)")
            
            print(f"\nDetailed Metrics:")
            metrics_count = 0
            for metric, value in result['metrics'].items():
                if isinstance(value, (int, float)):
                    metric_name = metric.replace('_', ' ').title()
                    print(f"  {metric_name:30}: {value*100:.2f}%")
                    metrics_count += 1
            
            print(f"\nToken Statistics:")
            counts = result['token_counts']
            print(f"  File 1: {counts['code1']} tokens ({counts['unique_types1']} unique types)")
            print(f"  File 2: {counts['code2']} tokens ({counts['unique_types2']} unique types)")
            print(f"  Common token types: {counts['common_types']}")
            
            lengths = result['code_lengths']
            print(f"\nCode Size:")
            print(f"  File 1: {lengths['code1_lines']} lines, {lengths['code1_chars']} characters")
            print(f"  File 2: {lengths['code2_lines']} lines, {lengths['code2_chars']} characters")
            
            # variable analysis
            if 'variable_patterns' in result:
                vars_data = result['variable_patterns']
                print(f"\nVariable Analysis:")
                print(f"  Variables: {vars_data['count1']} vs {vars_data['count2']}")
                print(f"  Common variable names: {vars_data['common_count']}")
                if vars_data['common_names']:
                    print(f"  Examples: {', '.join(vars_data['common_names'][:5])}")
            
            if result['matched_sections']:
                print(f"\nMatching Sections ({len(result['matched_sections'])} found):")
                for i, match in enumerate(result['matched_sections'][:5], 1):
                    print(f"  Match {i}: {match['length']} tokens")
                    print(f"    Lines {match['line_numbers']['start1']}-{match['line_numbers']['end1']} ↔ Lines {match['line_numbers']['start2']}-{match['line_numbers']['end2']}")
                    sample = ' '.join(match['token_texts'][:5])
                    if len(match['token_texts']) > 5:
                        sample += ' ...'
                    print(f"    Sample: {sample}")
                
                if len(result['matched_sections']) > 5:
                    print(f"    ... and {len(result['matched_sections']) - 5} more matches")
            else:
                print(f"\nNo matching sections found.")
            
            # token frequencies
            print(f"\nMost Common Tokens:")
            freq1 = result['token_frequencies']['code1']
            freq2 = result['token_frequencies']['code2']
            print(f"  File 1: {', '.join([f'{k}({v})' for k, v in list(freq1.items())[:5]])}")
            print(f"  File 2: {', '.join([f'{k}({v})' for k, v in list(freq2.items())[:5]])}")
        
        output_base = args.output
        if output_base.endswith('.json') or output_base.endswith('.html') or output_base.endswith('.txt'):
            output_base = os.path.splitext(output_base)[0]
        
        print(f"\nSaving reports...")
        
        if args.format in ['json', 'all']:
            save_json_report(result, f"phase1/results/similarity_report_Phase1.json")
        if args.format in ['html', 'all'] or args.visual:
            save_html_report(analyzer, result, filenames[0], filenames[1], f"phase1/results/similarity_report_Phase1.html")
        if args.format in ['text', 'all']:
            save_text_report(result, filenames[0], filenames[1], f"phase1/results/similarity_report_Phase1.txt")
        
        print(f"\nAnalysis complete!")
        print(f"Overall similarity: {result['overall_similarity']:.1f}%")
        
    else:
        print("Error: For pairwise analysis exactly 2 files are required.")
        print("For matrix analysis, use --matrix flag with at least 2 files.")
        sys.exit(1)


if __name__ == "__main__":
    main()