"""
Direct execution file for phase 3
Run with command: python -m phase3.main
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analyzer.cfg_analyzer import Phase3CFGSimilarity, CFGAnalyzer
from visualizer.cfg_visualizer import visualize_cfg, generate_cfg_report
from integration.phase_integration import Phase3Integration


def read_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Control Flow Graph (CFG) based similarity analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m phase3.main -f1 code1.py -f2 code2.py
  python -m phase3.main -c1 "for i in range(10): print(i)" -c2 "for j in range(5): print(j)"
  python -m phase3.main --phase1 phase1.json --phase2 phase2.json --code1 ... --code2 ...
        """
    )

    parser.add_argument('-f1', '--file1', help='Path to first code file')
    parser.add_argument('-f2', '--file2', help='Path to second code file')
    parser.add_argument('-c1', '--code1', help='Text of first code')
    parser.add_argument('-c2', '--code2', help='Text of second code')

    parser.add_argument('--phase1', help='Phase 1 results file (JSON)')
    parser.add_argument('--phase2', help='Phase 2 results file (JSON)')

    parser.add_argument('-o', '--output', default='phase3_report.json',
                       help='Output report file (default: phase3_report.json)')
    parser.add_argument('-t', '--threshold', type=float, default=0.7,
                       help='Detection threshold (default: 0.7)')
    parser.add_argument('--language', '-l', default='python',
                       choices=['python', 'java', 'cpp'],
                       help='Code language (default: python)')
    parser.add_argument('--config', help='Phase 3 config file')

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize CFG')
    parser.add_argument('--dot', help='Generate DOT file for graph')

    args = parser.parse_args()

    code1, code2 = None, None

    if args.file1:
        code1 = read_file(args.file1)
    elif args.code1:
        code1 = args.code1

    if args.file2:
        code2 = read_file(args.file2)
    elif args.code2:
        code2 = args.code2

    if not code1 or not code2:
        print("Please specify input codes.")
        parser.print_help()
        sys.exit(1)

    print("=" * 60)
    print("Phase 3: Control Flow Graph (CFG) Analysis")
    print("=" * 60)

    phase1_results = None
    phase2_results = None

    if args.phase1:
        try:
            with open(args.phase1, 'r', encoding='utf-8') as f:
                phase1_results = json.load(f)
            print(f"Phase 1 results loaded")
        except Exception as e:
            print(f"Error loading phase 1: {e}")

    if args.phase2:
        try:
            with open(args.phase2, 'r', encoding='utf-8') as f:
                phase2_results = json.load(f)
            print(f"Phase 2 results loaded")
        except Exception as e:
            print(f"Error loading phase 2: {e}")

    print("\nRunning CFG analysis...")

    if phase1_results and phase2_results:
        print("Integrated analysis (Phase 1 + 2 + 3)...")
        analyzer = Phase3CFGSimilarity(args.config)
        results = analyzer.analyze_code_pair(
            code1, code2, phase1_results, phase2_results
        )
    else:
        print("Independent CFG analysis...")
        analyzer = CFGAnalyzer(args.language, args.config)
        results = analyzer.analyze_code_pair(code1, code2)

    if args.threshold != 0.7:
        analyzer.config['plagiarism_threshold'] = args.threshold
        results['threshold_used'] = args.threshold

        if 'combined_similarity_score' in results:
            score = results['combined_similarity_score'] / 100
        else:
            score = results.get('cfg_similarity_score', 0) / 100

        results['is_plagiarism_suspected'] = score >= args.threshold

    print("\nGenerating report...")

    if args.verbose:
        print(generate_cfg_report(results))
    else:
        report = generate_cfg_report(results)
        print(report[:1000] + "..." if len(report) > 1000 else report)

    if args.visualize:
        print("\nVisualizing CFG...")

        cfg_analyzer = CFGAnalyzer(args.language)

        ast1, ast2 = cfg_analyzer._get_asts_from_phase2(code1, code2)
        cfg1 = cfg_analyzer.build_cfg_from_ast(ast1)
        cfg2 = cfg_analyzer.build_cfg_from_ast(ast2)

        print("\nFirst code CFG:")
        print(visualize_cfg(cfg1, max_nodes=20))

        print("\nSecond code CFG:")
        print(visualize_cfg(cfg2, max_nodes=20))

    if args.dot:
        from visualizer.cfg_visualizer import generate_cfg_dot_file

        cfg_analyzer = CFGAnalyzer(args.language)
        ast1, _ = cfg_analyzer._get_asts_from_phase2(code1, code2)
        cfg = cfg_analyzer.build_cfg_from_ast(ast1)

        dot_file = args.dot if args.dot.endswith('.dot') else args.dot + '.dot'
        generate_cfg_dot_file(cfg, dot_file)
        print(f"DOT file saved in {dot_file}.")

    output_json = args.output if args.output.endswith('.json') else args.output + '.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved in {output_json}.")

    print("\n" + "=" * 60)
    print("Final Result:")

    if 'combined_similarity_score' in results:
        score = results['combined_similarity_score']
        print(f"Combined score (Phase 1+2+3): {score:.2f}%")
    else:
        score = results.get('cfg_similarity_score', 0)
        print(f"Behavioral score (CFG): {score:.2f}%")

    threshold = results.get('threshold_used', 0.7) * 100
    is_plagiarism = results.get('is_plagiarism_suspected', False)

    if is_plagiarism:
        print(f"Verdict: Similar (Possible plagiarism)")
        print(f"   Explanation: Score above threshold {threshold:.0f}%")
    else:
        print(f"Verdict: Not similar")
        print(f"Explanation: Score below threshold {threshold:.0f}%")

    print("=" * 60)


if __name__ == '__main__':
    main()