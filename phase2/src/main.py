"""
Direct execution file for Phase 2
Run with command: python -m phase2
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from .similarity import calculate_ast_similarity
from .bridge import run_phase1_simple, load_phase1_results


def read_file(file_path: str) -> str:
    """Read file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


def detect_language(filename: str) -> str:
    """Detect language from file extension"""
    ext = Path(filename).suffix.lower()

    extensions_map = {
        '.py': 'python',
        '.java': 'java',
        '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', '.hpp': 'cpp', '.h': 'cpp',
        '.c': 'c',
        '.js': 'javascript',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.cs': 'csharp',
    }

    return extensions_map.get(ext, 'python')  # Default to Python


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: AST-based Similarity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f1 code1.py -f2 code2.py
  %(prog)s -c1 "def sum(a,b): return a+b" -c2 "def add(x,y): return x+y"
  %(prog)s --phase1-results results_phase1.json --file1 code1.py --file2 code2.py
  %(prog)s --integrate -f1 code1.py -f2 code2.py  (Automatically run both phases)
        """
    )

    # Input options
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('-f1', '--file1', help='Path to first code file')
    input_group.add_argument('-f2', '--file2', help='Path to second code file')
    input_group.add_argument('-c1', '--code1', help='First code text')
    input_group.add_argument('-c2', '--code2', help='Second code text')

    # Phase 1 options
    phase1_group = parser.add_argument_group('Phase 1 Integration')
    phase1_group.add_argument('--phase1-results', help='Phase 1 results file (JSON)')
    phase1_group.add_argument('--phase1-config', help='Phase 1 config file')
    phase1_group.add_argument('--integrate', action='store_true',
                              help='Automatically run Phase 1 and Phase 2')

    # Phase 2 options
    phase2_group = parser.add_argument_group('Phase 2 Settings')
    phase2_group.add_argument('-o', '--output', default='phase2_report.json',
                              help='Output report file (default: phase2_report.json)')
    phase2_group.add_argument('-t', '--threshold', type=float, default=0.65,
                              help='Plagiarism detection threshold (default: 0.65)')
    phase2_group.add_argument('--language', '-l', default='auto',
                              choices=['auto', 'python', 'java', 'cpp'],
                              help='Code language (default: auto)')
    phase2_group.add_argument('--config', help='Phase 2 config file')

    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--verbose', '-v', action='store_true',
                              help='Show detailed output')
    output_group.add_argument('--no-json', action='store_true',
                              help='Do not save JSON output')
    output_group.add_argument('--text-report', action='store_true',
                              help='Create separate text report')

    args = parser.parse_args()

    code1, code2 = None, None
    language = args.language

    if args.file1:
        code1 = read_file(args.file1)
        if language == 'auto':
            language = detect_language(args.file1)
    elif args.code1:
        code1 = args.code1

    if args.file2:
        code2 = read_file(args.file2)
        if language == 'auto' and not code1:
            language = detect_language(args.file2)
    elif args.code2:
        code2 = args.code2

    if not code1 or not code2:
        print(" Please specify input codes.")
        parser.print_help()
        sys.exit(1)

    print("=" * 60)
    print("Phase 2: Abstract Syntax Tree (AST) Similarity Analysis")
    print("=" * 60)

    results = None
    phase1_results = None

    # Load Phase 1 results if available
    if args.phase1_results:
        print(" Loading Phase 1 results...")
        try:
            phase1_results = load_phase1_results(args.phase1_results)
            print(f" Phase 1 results loaded (Score: {phase1_results.get('overall_similarity', 0):.1f}%)")
        except Exception as e:
            print(f" Error loading Phase 1 results: {e}")

    # Automatically run Phase 1
    elif args.integrate:
        print(" Running Phase 1 automatically...")
        phase1_results = run_phase1_simple(code1, code2, args.phase1_config)
        if phase1_results:
            print(f" Phase 1 executed (Score: {phase1_results.get('overall_similarity', 0):.1f}%)")
        else:
            print(" Phase 1 not executed. Only Phase 2 will run.")

    # Run analysis
    if phase1_results:
        print("\n Running integrated analysis (Phase 1 + Phase 2)...")
        from .similarity import integrate_with_phase1
        results = integrate_with_phase1(
            phase1_results, code1, code2,
            language, args.config
        )
    else:
        print("\n Running structural analysis (AST)...")
        results = calculate_ast_similarity(
            code1, code2, language, args.config
        )

    if args.threshold != 0.65:
        ast_score = results.get('ast_similarity_score', 0) / 100
        if 'combined_similarity_score' in results:
            combined_score = results.get('combined_similarity_score', 0) / 100
            results['is_plagiarism_suspected'] = combined_score >= args.threshold
        else:
            results['is_plagiarism_suspected'] = ast_score >= args.threshold
        results['threshold_used'] = args.threshold

    # Generate report
    print("\n Generating report...")

    if args.verbose:
        from .visualizer import visualize_ast_comparison
        print(visualize_ast_comparison(results))

    # Save results
    if not args.no_json:
        output_json = args.output if args.output.endswith('.json') else args.output + '.json'
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f" Results saved to {output_json}.")

    # Text report
    if args.text_report or args.verbose:
        from .similarity import generate_text_report
        report = generate_text_report(results)

        if args.text_report:
            txt_file = args.output.replace('.json', '.txt')
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f" Text report saved to {txt_file}.")

        if not args.verbose:
            print(report)

    # Display final result
    print("\n" + "=" * 60)
    print(" Final Result:")

    if 'combined_similarity_score' in results:
        score = results['combined_similarity_score']
        print(f" Combined Score: {score:.2f}%")
    else:
        score = results.get('ast_similarity_score', 0)
        print(f" Structural Score: {score:.2f}%")

    threshold = results.get('threshold_used', 0.65) * 100
    is_plagiarism = results.get('is_plagiarism_suspected', False)

    if is_plagiarism:
        print(f" Detection: Similar (Possible Plagiarism) - Above threshold {threshold:.0f}%")
    else:
        print(f" Detection: Not Similar - Below threshold {threshold:.0f}%")

    print("=" * 60)


if __name__ == '__main__':
    main()