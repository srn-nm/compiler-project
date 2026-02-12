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
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ماتریس شباهت کد - تحلیل فاز ۱</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700&display=swap');
            
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Vazirmatn', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #0f172a;
                padding: 30px 20px;
                color: #e2e8f0;
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 1600px;
                margin: 0 auto;
                background: #1e293b;
                padding: 40px;
                border-radius: 16px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                border: 1px solid #334155;
            }}
            
            h1 {{
                font-size: 2em;
                font-weight: 700;
                color: #60a5fa;
                margin-bottom: 20px;
                text-align: center;
                padding-bottom: 15px;
                border-bottom: 2px solid #334155;
            }}
            
            h2 {{
                color: #94a3b8;
                margin: 30px 0 20px 0;
                padding-bottom: 10px;
                border-bottom: 1px solid #334155;
                font-weight: 600;
            }}
            
            .stats {{
                background: #0f172a;
                padding: 25px;
                border-radius: 12px;
                margin: 20px 0;
                border: 1px solid #334155;
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                text-align: center;
            }}
            
            .stat-item {{
                padding: 15px;
                background: #1e293b;
                border-radius: 8px;
                border: 1px solid #334155;
            }}
            
            .stat-value {{
                font-size: 36px;
                font-weight: 700;
                color: #60a5fa;
                margin-bottom: 5px;
            }}
            
            .stat-label {{
                color: #94a3b8;
                font-size: 14px;
            }}
            
            .matrix-container {{
                margin: 30px 0;
                overflow-x: auto;
                direction: ltr;
                text-align: left;
                background: #0f172a;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #334155;
            }}
            
            .matrix {{
                display: inline-block;
            }}
            
            .matrix-row {{
                display: flex;
                align-items: center;
                margin: 4px 0;
            }}
            
            .matrix-label {{
                width: 70px;
                padding: 10px;
                font-weight: 600;
                color: #94a3b8;
                font-size: 14px;
                text-align: center;
            }}
            
            .matrix-cell {{
                width: 70px;
                height: 70px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 2px;
                border-radius: 8px;
                color: white;
                font-weight: 700;
                font-size: 16px;
                transition: transform 0.2s, box-shadow 0.2s;
                box-shadow: 0 2px 4px rgba(0,0,0,0.5);
                text-shadow: 0 1px 2px rgba(0,0,0,0.5);
            }}
            
            .matrix-cell:hover {{
                transform: scale(1.05);
                box-shadow: 0 4px 12px rgba(0,0,0,0.8);
                border: 1px solid rgba(255,255,255,0.2);
            }}
            
            .comparisons-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            
            .comparison-card {{
                background: #0f172a;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #334155;
                transition: all 0.2s;
            }}
            
            .comparison-card:hover {{
                border-color: #60a5fa;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(96, 165, 250, 0.2);
                background: #0a0f1a;
            }}
            
            .comparison-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid #334155;
            }}
            
            .file-badge {{
                background: #1e293b;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 600;
                color: #60a5fa;
                border: 1px solid #334155;
            }}
            
            .similarity-circle {{
                width: 70px;
                height: 70px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 20px;
                background: #0f172a;
                border: 2px solid;
                margin: 0 auto;
                box-shadow: 0 2px 8px rgba(0,0,0,0.5);
            }}
            
            .token-stats {{
                display: flex;
                justify-content: space-between;
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #334155;
                color: #94a3b8;
                font-size: 13px;
            }}
            
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 30px;
                border-top: 1px solid #334155;
                color: #64748b;
            }}
            
            .badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                background: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
            }}
            
            .similarity-legend {{
                display: flex;
                align-items: center;
                gap: 30px;
                margin-top: 20px;
                padding: 20px;
                background: #0f172a;
                border-radius: 8px;
                border: 1px solid #334155;
                flex-wrap: wrap;
            }}
            
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 13px;
                color: #94a3b8;
            }}
            
            .legend-color {{
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 1px solid rgba(0,0,0,0.3);
            }}
            
            .filename-list {{
                background: #0f172a;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                border: 1px solid #334155;
                color: #cbd5e1;
                font-size: 14px;
            }}
            
            .phase-badge {{
                background: #2563eb;
                padding: 8px 16px;
                border-radius: 8px;
                font-weight: 600;
                display: inline-block;
                color: white;
                border: 1px solid #3b82f6;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }}
            
            .filename-highlight {{
                color: #60a5fa;
                font-weight: 600;
            }}
            
            .detail-text {{
                color: #cbd5e1;
            }}
            
            @media (max-width: 768px) {{
                .container {{
                    padding: 20px;
                }}
                
                .matrix-label,
                .matrix-cell {{
                    width: 50px;
                    height: 50px;
                    font-size: 14px;
                }}
                
                .stats {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 ماتریس شباهت کد - تحلیل فاز ۱</h1>
            
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value">{results['num_files']}</div>
                    <div class="stat-label">فایل تحلیل شده</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{len(results['comparisons'])}</div>
                    <div class="stat-label">مقایسه انجام شده</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">
                        {max([max(row) for row in results['similarity_matrix']])*100:.0f}%
                    </div>
                    <div class="stat-label">بیشترین شباهت</div>
                </div>
            </div>
            
            <div class="filename-list">
                <span class="filename-highlight">📁 فایل‌های ورودی:</span>
                <span style="color: #cbd5e1;"> {', '.join(filenames)}</span>
            </div>
            
            <h2>🔗 ماتریس شباهت</h2>
            <div class="matrix-container">
                <div class="matrix">
                    <div class="matrix-row">
                        <div class="matrix-label"></div>
    """
    
    for i in range(results['num_files']):
        html += f'<div class="matrix-label" style="color: #94a3b8;">فایل {i+1}</div>'
    html += '</div>'
    
    matrix = results['similarity_matrix']
    for i in range(results['num_files']):
        html += f'<div class="matrix-row">'
        html += f'<div class="matrix-label" style="color: #94a3b8;">فایل {i+1}</div>'
        
        for j in range(results['num_files']):
            val = matrix[i][j]
            if val > 0.7:
                bg_color = '#10b981'  # سبز تیره
            elif val > 0.4:
                bg_color = '#f59e0b'  # نارنجی تیره
            else:
                bg_color = '#ef4444'  # قرمز تیره
            html += f'<div class="matrix-cell" style="background-color: {bg_color};">{val*100:.0f}%</div>'
        
        html += '</div>'
    
    html += f"""
                </div>
            </div>
            
            <div class="similarity-legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #10b981;"></div>
                    <span>شباهت بالا (&gt;70%)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #f59e0b;"></div>
                    <span>شباهت متوسط (40-70%)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ef4444;"></div>
                    <span>شباهت کم (&lt;40%)</span>
                </div>
                <div class="legend-item">
                    <span class="badge">⚡ آستانه: 65%</span>
                </div>
            </div>
            
            <h2>📋 مقایسه‌های دقیق</h2>
            <div class="comparisons-grid">
    """
    
    for comp in results['comparisons'][:12]: 
        similarity = comp['similarity'] * 100
        if similarity > 70:
            color = '#10b981'
        elif similarity > 40:
            color = '#f59e0b'
        else:
            color = '#ef4444'
        
        html += f'''
                <div class="comparison-card">
                    <div class="comparison-header">
                        <span class="file-badge">فایل {comp['file1']+1}</span>
                        <span style="color: #64748b;">↔</span>
                        <span class="file-badge">فایل {comp['file2']+1}</span>
                    </div>
                    <div style="text-align: center;">
                        <div class="similarity-circle" style="border-color: {color}; color: {color};">
                            {similarity:.0f}%
                        </div>
                        <div style="margin-top: 10px; font-weight: 600; color: #e2e8f0;">
                            شباهت کلی
                        </div>
                    </div>
                    <div style="margin-top: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span style="color: #94a3b8;">توکن‌های فایل {comp['file1']+1}:</span>
                            <span style="color: #60a5fa; font-weight: 600;">{comp['details']['token_counts']['code1']:,}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span style="color: #94a3b8;">توکن‌های فایل {comp['file2']+1}:</span>
                            <span style="color: #60a5fa; font-weight: 600;">{comp['details']['token_counts']['code2']:,}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span style="color: #94a3b8;">بخش‌های مشابه:</span>
                            <span style="color: #60a5fa; font-weight: 600;">{len(comp['details']['matched_sections'])}</span>
                        </div>
                    </div>
                    <div class="token-stats">
                        <span>🎯 {comp['details']['token_counts']['common_types']} نوع مشترک</span>
                        <span>📊 {comp['details'].get('overall_similarity', comp['similarity']*100):.1f}%</span>
                    </div>
                </div>
        '''
    
    html += f'''
            </div>
    '''
    
    if len(results['comparisons']) > 12:
        html += f'''
            <div style="text-align: center; margin-top: 20px; padding: 15px; background: #0f172a; border-radius: 8px; border: 1px solid #334155;">
                <span style="color: #94a3b8;">✨ و {len(results['comparisons']) - 12} مقایسه دیگر ...</span>
            </div>
        '''
    
    html += f'''
            <div style="display: flex; align-items: center; gap: 15px; margin-top: 30px; padding: 20px; background: #0f172a; border-radius: 8px; border: 1px solid #334155;">
                <span class="phase-badge">تحلیل فاز ۱</span>
                <span style="color: #94a3b8;">
                    ⏱️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </span>
                <span style="color: #94a3b8; margin-right: auto;">
                    🎯 Token-based Similarity
                </span>
            </div>
            
            <div class="footer">
                <p>تولید شده توسط TokenSimilarityAnalyzer - فاز ۱</p>
                <p style="font-size: 12px; margin-top: 10px; color: #475569;">
                    تحلیل مبتنی بر توکن • {results['num_files']} فایل • {len(results['comparisons'])} جفت
                </p>
            </div>
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