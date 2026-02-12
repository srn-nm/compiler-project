import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analyzer.cfg_analyzer import Phase3CFGSimilarity, CFGAnalyzer
from visualizer.cfg_visualizer import visualize_cfg, generate_cfg_report, generate_cfg_dot_file
from utils.helpers import save_json, load_json, format_decision


def read_file(file_path: str) -> str:
    """Read code file safely"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()
    except Exception as e:
        print(f" Error reading file {file_path}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Control Flow Graph (CFG) Similarity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic CFG analysis
  python -m phase3.main -f1 code1.py -f2 code2.py
  
  # Integrated analysis with Phase 1 & 2
  python -m phase3.main --phase1 phase1.json --phase2 phase2.json -f1 code1.py -f2 code2.py
  
  # Full three-phase analysis
  python -m phase3.main --full -f1 code1.py -f2 code2.py
  
  # Visualization
  python -m phase3.main -f1 code1.py -f2 code2.py --visualize --dot cfg.dot
        """
    )

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('-f1', '--file1', help='First code file')
    input_group.add_argument('-f2', '--file2', help='Second code file')
    input_group.add_argument('-c1', '--code1', help='First code text')
    input_group.add_argument('-c2', '--code2', help='Second code text')

    phase_group = parser.add_argument_group('Phase Integration')
    phase_group.add_argument('--phase1', help='Phase 1 results file (JSON)')
    phase_group.add_argument('--phase2', help='Phase 2 results file (JSON)')
    phase_group.add_argument('--full', action='store_true', help='Run complete three-phase analysis')

    cfg_group = parser.add_argument_group('CFG Settings')
    cfg_group.add_argument('-o', '--output', default='phase3_report.json', help='Output file')
    cfg_group.add_argument('-t', '--threshold', type=float, default=0.7, help='Detection threshold (0.0-1.0)')
    cfg_group.add_argument('--language', '-l', default='python', choices=['python', 'java', 'cpp'], help='Language')
    cfg_group.add_argument('--config', help='Config file')

    viz_group = parser.add_argument_group('Visualization')
    viz_group.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    viz_group.add_argument('--visualize', action='store_true', help='Visualize CFG in terminal')
    viz_group.add_argument('--dot', help='Generate DOT file for Graphviz')

    args = parser.parse_args()

    code1, code2 = None, None
    file1_name, file2_name = "code1", "code2"

    if args.file1:
        code1 = read_file(args.file1)
        file1_name = Path(args.file1).name
    elif args.code1:
        code1 = args.code1

    if args.file2:
        code2 = read_file(args.file2)
        file2_name = Path(args.file2).name
    elif args.code2:
        code2 = args.code2

    if not code1 or not code2:
        print(" Please specify input codes.")
        parser.print_help()
        sys.exit(1)

    print("\n" + "=" * 70)
    print("                 PHASE 3: CFG ANALYSIS")
    print("=" * 70)
    print(f" File 1: {file1_name}")
    print(f" File 2: {file2_name}")
    print(f" Language: {args.language}")
    print("-" * 70)

    phase1_results = None
    phase2_results = None

    if args.full:
        print("\n Running complete three-phase analysis...")
        from integration.phase_integration import run_complete_analysis
        results = run_complete_analysis(code1, code2, args.output)
        print("\n Analysis complete!")
        
    else:
        if args.phase1:
            try:
                phase1_results = load_json(args.phase1)
                print(f" Phase 1 results loaded")
            except Exception as e:
                print(f" Error loading phase 1: {e}")

        if args.phase2:
            try:
                phase2_results = load_json(args.phase2)
                print(f" Phase 2 results loaded")
            except Exception as e:
                print(f" Error loading phase 2: {e}")

        print("\n Running CFG analysis...")

        if phase1_results and phase2_results:
            print("   Mode: Integrated (Phase 1 + 2 + 3)")
            analyzer = Phase3CFGSimilarity(args.config)
            results = analyzer.analyze_code_pair(code1, code2, phase1_results, phase2_results)
        else:
            print("   Mode: CFG only")
            analyzer = CFGAnalyzer(language=args.language, config_path=args.config)
            results = analyzer.analyze_code_pair(code1, code2)

        if args.threshold != 0.7:
            if 'combined_similarity_score' in results:
                score = results['combined_similarity_score'] / 100
            else:
                score = results.get('cfg_similarity_score', 0) / 100
            results['is_plagiarism_suspected'] = score >= args.threshold
            results['threshold_used'] = args.threshold

    print("\n ANALYSIS RESULTS")
    print("-" * 70)

    if 'combined_similarity_score' in results:
        combined = results['combined_similarity_score']
        print(f" Combined Score (3 phases): {combined:.2f}%")
        if 'individual_scores' in results:
            scores = results['individual_scores']
            print(f"   • Token: {scores.get('token', 0):.1f}%")
            print(f"   • AST:   {scores.get('ast', 0):.1f}%")
            print(f"   • CFG:   {scores.get('cfg', 0):.1f}%")
    else:
        cfg_score = results.get('cfg_similarity_score', 0)
        print(f" CFG Similarity Score: {cfg_score:.2f}%")

    # Decision
    is_plagiarism = results.get('is_plagiarism_suspected', False)
    threshold = results.get('threshold_used', 0.7) * 100
    decision_text, emoji = format_decision(is_plagiarism, threshold)
    
    print(f"\n  VERDICT:")
    print(f"   • Threshold: {threshold:.0f}%")
    print(f"   • Result: {emoji} {decision_text}")

    if 'confidence' in results:
        print(f"   • Confidence: {results['confidence']:.1f}%")

    if args.verbose and not args.full:
        print("\n" + "=" * 70)
        print("🔍 DETAILED CFG REPORT")
        print("=" * 70)
        print(generate_cfg_report(results))

    if args.visualize and not args.full:
        print("\n" + "=" * 70)
        print(" CFG VISUALIZATION")
        print("=" * 70)
        
        cfg_analyzer = CFGAnalyzer(language=args.language)
        
        if phase2_results and 'ast1_dict' in phase2_results and 'ast2_dict' in phase2_results:
            ast1 = phase2_results['ast1_dict']
            ast2 = phase2_results['ast2_dict']
        else:
            from analyzer.cfg_builder import create_mock_ast
            ast1 = create_mock_ast()
            ast2 = create_mock_ast()
        
        cfg1 = cfg_analyzer.build_cfg_from_ast(ast1)
        cfg2 = cfg_analyzer.build_cfg_from_ast(ast2)
        
        print(f"\n {file1_name}:")
        print(visualize_cfg(cfg1, max_nodes=15))
        
        print(f"\n {file2_name}:")
        print(visualize_cfg(cfg2, max_nodes=15))

    if args.dot and not args.full:
        cfg_analyzer = CFGAnalyzer(language=args.language)
        
        if phase2_results and 'ast1_dict' in phase2_results:
            ast1 = phase2_results['ast1_dict']
        else:
            from analyzer.cfg_builder import create_mock_ast
            ast1 = create_mock_ast()
            
        cfg = cfg_analyzer.build_cfg_from_ast(ast1)
        
        dot_file = args.dot if args.dot.endswith('.dot') else args.dot + '.dot'
        generate_cfg_dot_file(cfg, dot_file)
        print(f"\n DOT file: {dot_file}")

    if not args.full:
        output_file = save_json(results, args.output)
        print(f"\n Results saved: {output_file}")

    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()