import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase3.analyzer.cfg_analyzer import Phase3CFGSimilarity, CFGAnalyzer
from phase3.visualizer.cfg_visualizer import visualize_cfg, generate_cfg_report, generate_cfg_dot_file
from phase3.visualizer.html_visualizer import generate_cfg_html, generate_cfg_comparison_html
from phase3.integration.phase_integration import Phase3Integration, run_complete_analysis
from phase3.utils.helpers import save_json, load_json, format_percentage, format_decision, create_summary


def read_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()
    except Exception as e:
        print(f" Error reading file {file_path}: {e}")
        sys.exit(1)


def get_ast_from_phase2(phase2_results: dict) -> tuple:
    
    ast1 = None
    ast2 = None
    
    if phase2_results:
        # Try all possible keys from Phase 2
        ast1 = (phase2_results.get('ast1_dict') or 
                phase2_results.get('ast1') or 
                phase2_results.get('ast_1'))
        
        ast2 = (phase2_results.get('ast2_dict') or 
                phase2_results.get('ast2') or 
                phase2_results.get('ast_2'))
        
        if ast1 and ast2:
            print(f"    Found AST dictionaries in Phase 2 results")
            print(f"      • ast1_dict: {type(ast1).__name__}")
            print(f"      • ast2_dict: {type(ast2).__name__}")
    
    return ast1, ast2


def load_phase2_results(phase2_file: str) -> dict:
    try:
        with open(phase2_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if 'ast1_dict' in results and 'ast2_dict' in results:
            print(f"    Phase 2 results loaded with AST dictionaries")
        
        return results
    except Exception as e:
        print(f"    Error loading Phase 2 results: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Control Flow Graph (CFG) Similarity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic CFG analysis (with built-in AST parser)
  python -m phase3.main -f1 code1.py -f2 code2.py
  
  # With Phase 2 AST dictionaries (RECOMMENDED)
  python -m phase3.main --phase2 phase2/results/phase2_report.json -f1 code1.py -f2 code2.py
  
  # With HTML visualization
  python -m phase3.main --phase2 phase2_report.json -f1 code1.py -f2 code2.py --html --html-compare
  
  # Full three-phase analysis
  python -m phase3.main --full -f1 code1.py -f2 code2.py --html-compare
  
  # Generate DOT file for Graphviz
  python -m phase3.main --phase2 phase2_report.json -f1 code1.py -f2 code2.py --dot cfg.dot
        """
    )

    # Input options
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('-f1', '--file1', help='Path to first code file')
    input_group.add_argument('-f2', '--file2', help='Path to second code file')
    input_group.add_argument('-c1', '--code1', help='Text of first code')
    input_group.add_argument('-c2', '--code2', help='Text of second code')

    # Phase integration options
    phase_group = parser.add_argument_group('Phase Integration')
    phase_group.add_argument('--phase1', help='Phase 1 results file (JSON)')
    phase_group.add_argument('--phase2', help='Phase 2 results file (JSON) - RECOMMENDED for AST')
    phase_group.add_argument('--full', action='store_true', help='Run complete three-phase analysis')

    # CFG options
    cfg_group = parser.add_argument_group('CFG Settings')
    cfg_group.add_argument('-o', '--output', default='phase3/results/phase3_report.json', help='Output file (JSON)')
    cfg_group.add_argument('-t', '--threshold', type=float, default=0.7, help='Detection threshold (0.0-1.0)')
    cfg_group.add_argument('--language', '-l', default='python', choices=['python', 'java', 'cpp'], help='Language')
    cfg_group.add_argument('--config', help='Config file')

    # Visualization options
    viz_group = parser.add_argument_group('Visualization')
    viz_group.add_argument('--verbose', '-v', action='store_true', help='Show detailed output in terminal')
    viz_group.add_argument('--visualize', action='store_true', help='Visualize CFG in terminal')
    viz_group.add_argument('--dot', help='Generate DOT file for Graphviz')
    viz_group.add_argument('--html', action='store_true', help='Generate interactive HTML visualization')
    viz_group.add_argument('--html-compare', action='store_true', help='Generate HTML comparison of both CFGs')
    viz_group.add_argument('--no-summary', action='store_true', help='Do not display summary')

    args = parser.parse_args()

    code1, code2 = None, None
    file1_name, file2_name = "code1.py", "code2.py"

    if args.file1:
        code1 = read_file(args.file1)
        file1_name = Path(args.file1).name
    elif args.code1:
        code1 = args.code1
        file1_name = "code1.txt"

    if args.file2:
        code2 = read_file(args.file2)
        file2_name = Path(args.file2).name
    elif args.code2:
        code2 = args.code2
        file2_name = "code2.txt"

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
    ast1 = None
    ast2 = None

    # Full three-phase analysis
    if args.full:
        print("\n Running complete three-phase analysis...")
        results = run_complete_analysis(code1, code2, args.output)
        
    else:
        # Load Phase 1 results if provided
        if args.phase1:
            try:
                phase1_results = load_json(args.phase1)
                score1 = phase1_results.get('overall_similarity', 0)
                print(f" Phase 1 results loaded - Score: {score1:.1f}%")
            except Exception as e:
                print(f" Error loading phase 1: {e}")

        # Load Phase 2 results if provided - THIS IS THE IMPORTANT PART
        if args.phase2:
            print("\n Loading Phase 2 results with AST dictionaries...")
            phase2_results = load_phase2_results(args.phase2)
            
            if phase2_results:
                # Extract AST dictionaries
                ast1, ast2 = get_ast_from_phase2(phase2_results)
                
                score2 = phase2_results.get('ast_similarity_score', 0)
                print(f"    Phase 2 score: {score2:.1f}%")
                
                if ast1 and ast2:
                    print(f"    AST dictionaries ready for CFG building")
                    print(f"      • ast1_dict: {len(str(ast1))} chars")
                    print(f"      • ast2_dict: {len(str(ast2))} chars")
                else:
                    print(f"    AST dictionaries not found in Phase 2 results")
            else:
                print(f"    Failed to load Phase 2 results")

        print("\n Running CFG analysis...")

        if phase1_results and phase2_results:
            print("   Mode: Integrated (Phase 1 + 2 + 3)")
            analyzer = Phase3CFGSimilarity(args.config)
            results = analyzer.analyze_code_pair(
                code1, code2, 
                phase1_results=phase1_results, 
                phase2_results=phase2_results
            )
        elif phase2_results:
            print("   Mode: CFG with Phase 2 AST (RECOMMENDED)")
            analyzer = Phase3CFGSimilarity(args.config)
            results = analyzer.analyze_code_pair(
                code1, code2, 
                phase2_results=phase2_results
            )
        else:
            print("   Mode: CFG only (with built-in AST parser)")
            analyzer = CFGAnalyzer(language=args.language, config_path=args.config)
            results = analyzer.analyze_code_pair(code1, code2)

        if args.threshold != 0.7:
            if 'combined_similarity_score' in results:
                score = results['combined_similarity_score'] / 100
            else:
                score = results.get('cfg_similarity_score', 0) / 100
            results['is_plagiarism_suspected'] = score >= args.threshold
            results['threshold_used'] = args.threshold

    if not args.no_summary:
        print("\n ANALYSIS RESULTS")
        print("-" * 70)

        # Combined/CFG score
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

        # Metrics
        if 'cfg_similarity_metrics' in results:
            metrics = results['cfg_similarity_metrics']
            print(f"\n Similarity Metrics:")
            for key, value in list(metrics.items())[:4]:
                if isinstance(value, float):
                    name = key.replace('_', ' ').title()
                    print(f"   • {name:20}: {value*100:6.2f}%")

        # Decision
        is_plagiarism = results.get('is_plagiarism_suspected', False)
        threshold = results.get('threshold_used', 0.7) * 100
        decision_text, emoji = format_decision(is_plagiarism, threshold)
        
        print(f"\n  VERDICT:")
        print(f"   • Threshold: {threshold:.0f}%")
        print(f"   • Result: {emoji} {decision_text}")

        if 'confidence' in results:
            print(f"   • Confidence: {results['confidence']:.1f}%")

        if 'similar_components' in results and results['similar_components']:
            comp_count = len(results['similar_components'])
            print(f"\n Similar Components Found: {comp_count}")

    if args.verbose and not args.full:
        print("\n" + "=" * 70)
        print(" DETAILED CFG REPORT")
        print("=" * 70)
        print(generate_cfg_report(results))

    if args.visualize and not args.full:
        print("\n" + "=" * 70)
        print(" CFG VISUALIZATION (TERMINAL)")
        print("=" * 70)
        
        cfg_analyzer = CFGAnalyzer(language=args.language)
        
        if ast1 and ast2:
            print("   Using AST dictionaries from Phase 2")
            cfg1 = cfg_analyzer.build_cfg_from_ast(ast1)
            cfg2 = cfg_analyzer.build_cfg_from_ast(ast2)
        else:
            print("   Using built-in AST parser")
            from phase3.analyzer.cfg_builder import create_mock_ast
            ast1 = create_mock_ast()
            ast2 = create_mock_ast()
            cfg1 = cfg_analyzer.build_cfg_from_ast(ast1)
            cfg2 = cfg_analyzer.build_cfg_from_ast(ast2)
        
        print(f"\n {file1_name}:")
        print(visualize_cfg(cfg1, max_nodes=15))
        
        print(f"\n {file2_name}:")
        print(visualize_cfg(cfg2, max_nodes=15))

    html_files = []
    
    if args.html and not args.full:
        print("\n Generating interactive HTML visualization...")
        
        cfg_analyzer = CFGAnalyzer(language=args.language)
        
        if ast1:
            print("   Using AST dictionary from Phase 2")
            cfg = cfg_analyzer.build_cfg_from_ast(ast1)
        else:
            print("   Using built-in AST parser")
            from phase3.analyzer.cfg_builder import create_mock_ast
            ast = create_mock_ast()
            cfg = cfg_analyzer.build_cfg_from_ast(ast)
        
        base_name = Path(args.output).stem
        html_file = f"phase3/results/phase3_report_cfg.html"
        
        generate_cfg_html(cfg, file1_name, html_file)
        html_files.append(html_file)
        print(f"   HTML: {html_file}")

    if args.html_compare and not args.full:
        print("\n Generating interactive HTML comparison...")
        
        cfg_analyzer = CFGAnalyzer(language=args.language)
        
        if ast1 and ast2:
            print("   Using AST dictionaries from Phase 2")
            cfg1 = cfg_analyzer.build_cfg_from_ast(ast1)
            cfg2 = cfg_analyzer.build_cfg_from_ast(ast2)
        else:
            print("   Using built-in AST parser")
            from phase3.analyzer.cfg_builder import create_mock_ast
            ast1 = create_mock_ast()
            ast2 = create_mock_ast()
            cfg1 = cfg_analyzer.build_cfg_from_ast(ast1)
            cfg2 = cfg_analyzer.build_cfg_from_ast(ast2)
        
        base_name = Path(args.output).stem
        html_file = f"phase3/results/phase3_report_comparison.html"
        
        generate_cfg_comparison_html(
            cfg1, cfg2, 
            file1_name, file2_name, 
            results, 
            html_file
        )
        html_files.append(html_file)
        print(f"   HTML: {html_file}")

    if args.dot and not args.full:
        cfg_analyzer = CFGAnalyzer(language=args.language)
        
        if ast1:
            cfg = cfg_analyzer.build_cfg_from_ast(ast1)
        else:
            from phase3.analyzer.cfg_builder import create_mock_ast
            ast = create_mock_ast()
            cfg = cfg_analyzer.build_cfg_from_ast(ast)
        
        dot_file = args.dot if args.dot.endswith('.dot') else args.dot + '.dot'
        generate_cfg_dot_file(cfg, dot_file)
        print(f"\n DOT file: {dot_file}")

    if not args.full:
        output_file = save_json(results, args.output)
        print(f"\n Results saved: {output_file}")

    if html_files:
        print(f"\n HTML Visualizations:")
        for html_file in html_files:
            print(f"   • {html_file}")
        print(f"\n   Open these files in your browser to view interactive CFG graphs.")

    if not args.no_summary:
        print("\n" + create_summary(results))
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()