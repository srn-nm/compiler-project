import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from .similarity import calculate_ast_similarity, integrate_with_phase1, generate_text_report
from .bridge import Phase1Phase2Bridge
from .report_generator import generate_phase2_html_report, generate_integrated_html_report


def read_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


def detect_language(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    extensions_map = {
        '.py': 'python', '.java': 'java',
        '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', 
    }
    return extensions_map.get(ext, 'python')


def save_json_report(results: dict, output_path: str):
    output_json = output_path if output_path.endswith('.json') else output_path + '.json'
    with open("phase2/results/phase2_report.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"JSON report: phase2/results/phase2_report.json")
    return "phase2/results/phase2_report.json"


def save_text_report(results: dict, output_path: str):
    """Save results as text report"""
    txt_file = output_path.replace('.json', '.txt')
    report = generate_text_report(results)
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Text report: {txt_file}")
    return txt_file


def save_html_report(results: dict, file1_name: str, file2_name: str, output_path: str, phase1_results: dict = None):
    if phase1_results:
        html_file = output_path.replace('.json', '_integrated.html')
        generate_integrated_html_report(phase1_results, results, file1_name, file2_name, html_file)
    else:
        html_file = output_path.replace('.json', '.html')
        generate_phase2_html_report(results, file1_name, file2_name, html_file)
    return html_file


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: AST-based Similarity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f1 code1.py -f2 code2.py
  %(prog)s -f1 code1.py -f2 code2.py --html-only
  %(prog)s -c1 "def sum(a,b): return a+b" -c2 "def add(x,y): return x+y"
  %(prog)s --phase1-results results_phase1.json -f1 code1.py -f2 code2.py
  %(prog)s -f1 code1.py -f2 code2.py --verbose --text-report
        """
    )

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('-f1', '--file1', help='Path to first code file')
    input_group.add_argument('-f2', '--file2', help='Path to second code file')
    input_group.add_argument('-c1', '--code1', help='First code text')
    input_group.add_argument('-c2', '--code2', help='Second code text')

    # Phase 1 options
    phase1_group = parser.add_argument_group('Phase 1 Integration')
    phase1_group.add_argument('--phase1-results', help='Phase 1 results file (JSON)')
    phase1_group.add_argument('--phase1-config', help='Phase 1 config file')

    # Phase 2 options
    phase2_group = parser.add_argument_group('Phase 2 Settings')
    phase2_group.add_argument('-o', '--output', default='phase2_report',
                              help='Output report file name (without extension)')
    phase2_group.add_argument('-t', '--threshold', type=float, default=0.65,
                              help='Plagiarism detection threshold (default: 0.65)')
    phase2_group.add_argument('--language', '-l', default='auto',
                              choices=['auto', 'python', 'java', 'cpp'],
                              help='Code language (default: auto)')
    phase2_group.add_argument('--config', help='Phase 2 config file')

    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--verbose', '-v', action='store_true', help='Show detailed output in terminal')
    output_group.add_argument('--no-json', action='store_true', help='Do not save JSON output')
    output_group.add_argument('--no-html', action='store_true', help='Do not save HTML output')
    output_group.add_argument('--html-only', action='store_true', help='Save only HTML report (no JSON)')
    output_group.add_argument('--text-report', action='store_true', help='Create text report')
    output_group.add_argument('--output-dir', default='.', help='Output directory for reports')

    args = parser.parse_args()

    code1, code2 = None, None
    language = args.language
    file1_name = "code1"
    file2_name = "code2"

    if args.file1:
        code1 = read_file(args.file1)
        file1_name = Path(args.file1).name
        if language == 'auto':
            language = detect_language(args.file1)
    elif args.code1:
        code1 = args.code1
        file1_name = "code1.txt"
    else:
        print("Error: Please specify input code 1 (-f1 or -c1)")
        parser.print_help()
        sys.exit(1)

    if args.file2:
        code2 = read_file(args.file2)
        file2_name = Path(args.file2).name
        if language == 'auto' and not args.file1:
            language = detect_language(args.file2)
    elif args.code2:
        code2 = args.code2
        file2_name = "code2.txt"
    else:
        print("Error: Please specify input code 2 (-f2 or -c2)")
        parser.print_help()
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / args.output

    print("\n" + "=" * 70)
    print("                 PHASE 2: AST SIMILARITY ANALYSIS")
    print("=" * 70)
    print(f"File 1: {file1_name}")
    print(f"File 2: {file2_name}")
    print(f"Language: {language}")
    print("-" * 70)

    phase1_results = None
    if args.phase1_results:
        print("\nLoading Phase 1 results...")
        try:
            bridge = Phase1Phase2Bridge(args.phase1_results)
            phase1_results = bridge.phase1_results
            summary = bridge.get_summary()
            print(f"Phase 1 loaded: {summary.get('token_similarity', 0):.1f}% similarity")
            print(f"     • Matching sections: {summary.get('matching_sections', 0)}")
            print(f"     • Common variables: {summary.get('common_variables', 0)}")
        except Exception as e:
            print(f"Error loading Phase 1 results: {e}")
            print("Continuing with Phase 2 only...")

    print("\nRunning analysis...")
    
    if phase1_results:
        print("     Mode: Integrated (Phase 1 + Phase 2)")
        results = integrate_with_phase1(
            phase1_results, code1, code2, language, args.config
        )
    else:
        print("     Mode: AST only")
        results = calculate_ast_similarity(code1, code2, language, args.config)

    if args.threshold != 0.65:
        if 'combined_similarity_score' in results:
            combined_score = results.get('combined_similarity_score', 0) / 100
            results['is_plagiarism_suspected'] = combined_score >= args.threshold
        else:
            ast_score = results.get('ast_similarity_score', 0) / 100
            results['is_plagiarism_suspected'] = ast_score >= args.threshold
        results['threshold_used'] = args.threshold

    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    if 'combined_similarity_score' in results:
        print(f"\n Combined Score: {results['combined_similarity_score']:.1f}%")
        print(f"     • Token: {results.get('token_similarity_score', 0):.1f}%")
        print(f"     • AST:   {results.get('ast_similarity_score', 0):.1f}%")
    else:
        print(f"\n AST Similarity Score: {results.get('ast_similarity_score', 0):.1f}%")

    # AST Metrics
    ast_metrics = results.get('phase2_details', {}).get('ast_similarity_metrics', 
                  results.get('ast_similarity_metrics', {}))
    if ast_metrics:
        print(f"\n AST Metrics:")
        for metric, value in ast_metrics.items():
            metric_name = {
                'structural_similarity': 'Structural',
                'node_type_similarity': 'Node Type',
                'subtree_similarity': 'Subtree',
                'depth_similarity': 'Depth'
            }.get(metric, metric)
            print(f"     • {metric_name}: {value*100:.1f}%")

    ast_stats = results.get('phase2_details', {}).get('ast_statistics', {})
    if ast_stats:
        stats1 = ast_stats.get('code1', {})
        stats2 = ast_stats.get('code2', {})
        print(f"\n AST Statistics:")
        print(f"     • Code 1: {stats1.get('total_nodes', 0)} nodes, depth {stats1.get('max_depth', 0)}")
        print(f"     • Code 2: {stats2.get('total_nodes', 0)} nodes, depth {stats2.get('max_depth', 0)}")

    matched_nodes = results.get('phase2_details', {}).get('matched_nodes_count',
                    results.get('matched_nodes_count', 0))
    print(f"\n Similar Nodes: {matched_nodes}")

    # Decision
    threshold = results.get('threshold_used', 0.65) * 100
    is_plagiarism = results.get('is_plagiarism_suspected', False)
    print(f"\n Decision:")
    print(f"     • Threshold: {threshold:.0f}%")
    if is_plagiarism:
        print(f"     • Result: SIMILAR (Possible Plagiarism)")
    else:
        print(f"     • Result: NOT SIMILAR")

    if args.verbose:
        print("\n" + "=" * 70)
        print("VERBOSE OUTPUT")
        print("=" * 70)
        from .visualizer import visualize_ast_comparison
        print(visualize_ast_comparison(results))

    print("\n" + "=" * 70)
    print("SAVING REPORTS")
    print("=" * 70)

    saved_files = []

    # Save JSON
    if not args.no_json and not args.html_only:
        json_path = save_json_report(results, str(output_base))
        saved_files.append(json_path)

    # Save HTML
    if not args.no_html:
        html_path = save_html_report(
            results, 
            file1_name, 
            file2_name, 
            str("phase2/results/phase2_report") + ('.json' if not args.html_only else ''),
            phase1_results if phase1_results else None
        )
        saved_files.append(html_path)

    if args.text_report:
        txt_path = save_text_report(results, str(output_base) + '.json')
        saved_files.append(txt_path)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n  Output directory: {output_dir.absolute()}")
    for file in saved_files:
        print(f"     • {Path(file).name}")
    
    if 'combined_similarity_score' in results:
        final_score = results['combined_similarity_score']
    else:
        final_score = results.get('ast_similarity_score', 0)
    
    print(f"\n  Final Similarity Score: {final_score:.1f}%")
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()