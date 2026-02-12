"""
Phase 3 - Complete usage examples
Run: python -m phase3.example
"""

from analyzer.cfg_analyzer import Phase3CFGSimilarity, CFGAnalyzer
from visualizer.cfg_visualizer import visualize_cfg, generate_cfg_report
from integration.phase_integration import run_complete_analysis
from utils.helpers import format_percentage, format_decision


def example1_simple_functions():
    """Example 1: Simple functions with same behavior, different names"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Similar Behavior, Different Names")
    print("=" * 70)

    code1 = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""

    code2 = """
def compute_total(values):
    result = 0
    for value in values:
        result = result + value
    return result
"""

    analyzer = Phase3CFGSimilarity()
    results = analyzer.analyze_code_pair(code1, code2)

    score = results.get('cfg_similarity_score', 0)
    is_plagiarism = results.get('is_plagiarism_suspected', False)
    decision, emoji = format_decision(is_plagiarism, 70)

    print(f"CFG Similarity: {score:.2f}%")
    print(f"Verdict: {emoji} {decision}")
    
    if 'cfg_similarity_metrics' in results:
        metrics = results['cfg_similarity_metrics']
        print(f"\nMetrics:")
        print(f"  • Structural: {metrics.get('structural_similarity', 0)*100:.1f}%")
        print(f"  • Graph Edit: {metrics.get('graph_edit_similarity', 0)*100:.1f}%")
        print(f"  • Subgraph:   {metrics.get('subgraph_similarity', 0)*100:.1f}%")
    
    return results


def example2_control_structures():
    """Example 2: Complex control structures"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Complex Control Structures")
    print("=" * 70)

    code1 = """
def process_data(data):
    if not data:
        return []
    
    results = []
    for item in data:
        if item > 0:
            processed = transform(item)
            results.append(processed)
    
    return sorted(results)

def transform(x):
    return x * 2
"""

    code2 = """
def handle_items(items):
    if len(items) == 0:
        return []
    
    output = []
    for element in items:
        if element >= 0:
            modified = convert(element)
            output.append(modified)
    
    output.sort()
    return output

def convert(y):
    return y + y
"""

    analyzer = Phase3CFGSimilarity()
    results = analyzer.analyze_code_pair(code1, code2)

    print(generate_cfg_report(results))
    return results


def example3_different_algorithms():
    """Example 3: Different algorithms, different behavior"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Different Algorithms")
    print("=" * 70)

    code1 = """
def factorial_iterative(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
"""

    code2 = """
def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)
"""

    analyzer = Phase3CFGSimilarity()
    results = analyzer.analyze_code_pair(code1, code2)

    score = results.get('cfg_similarity_score', 0)
    is_plagiarism = results.get('is_plagiarism_suspected', False)
    decision, emoji = format_decision(is_plagiarism, 70)

    print(f"CFG Similarity: {score:.2f}%")
    print(f"Verdict: {emoji} {decision}")
    print(f"\nExplanation: Different control flow patterns")
    
    return results


def example4_complete_analysis():
    """Example 4: Complete three-phase analysis"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Complete Three-Phase Analysis")
    print("=" * 70)

    code1 = """
def find_max(numbers):
    if not numbers:
        return None
    
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    
    return max_val
"""

    code2 = """
def get_maximum(values):
    if len(values) == 0:
        return None
    
    maximum = values[0]
    for value in values:
        if value > maximum:
            maximum = value
    
    return maximum
"""

    print("Running all three phases (Token + AST + CFG)...")
    results = run_complete_analysis(code1, code2, output_file="complete_analysis.json")

    if 'combined_similarity_score' in results:
        print(f"\n🎯 Final Results:")
        print(f"   Combined Score: {results['combined_similarity_score']:.2f}%")
        
        if 'individual_scores' in results:
            scores = results['individual_scores']
            print(f"   • Token: {scores.get('token', 0):.1f}%")
            print(f"   • AST:   {scores.get('ast', 0):.1f}%")
            print(f"   • CFG:   {scores.get('cfg', 0):.1f}%")
        
        if 'confidence' in results:
            print(f"   • Confidence: {results['confidence']:.1f}%")
        
        is_plagiarism = results.get('is_plagiarism_suspected', False)
        decision, emoji = format_decision(is_plagiarism, 70)
        print(f"\n⚖️  Verdict: {emoji} {decision}")
        
        if 'recommendation' in results:
            print(f"\n💡 Recommendation: {results['recommendation']}")
    
    return results


def example5_cfg_visualization():
    """Example 5: CFG Visualization"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: CFG Visualization")
    print("=" * 70)

    code = """
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
"""

    analyzer = CFGAnalyzer()
    from analyzer.cfg_builder import create_mock_ast
    ast = create_mock_ast()
    cfg = analyzer.build_cfg_from_ast(ast)

    print(visualize_cfg(cfg, max_nodes=20))
    
    # Generate DOT file
    from visualizer.cfg_visualizer import generate_cfg_dot_file
    generate_cfg_dot_file(cfg, "binary_search.dot")
    print("📁 DOT file: binary_search.dot")
    
    return cfg


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("           PHASE 3 - COMPLETE EXAMPLES")
    print("=" * 70)

    results1 = example1_simple_functions()
    results2 = example2_control_structures()
    results3 = example3_different_algorithms()
    results4 = example4_complete_analysis()
    example5_cfg_visualization()

    # Summary
    print("\n" + "=" * 70)
    print("           SUMMARY")
    print("=" * 70)
    
    scores = [
        ("Simple functions", results1.get('cfg_similarity_score', 0)),
        ("Control structures", results2.get('cfg_similarity_score', 0)),
        ("Different algorithms", results3.get('cfg_similarity_score', 0)),
    ]
    
    for name, score in scores:
        print(f"  • {name:20}: {score:.1f}%")
    
    if 'combined_similarity_score' in results4:
        print(f"  • Complete analysis: {results4['combined_similarity_score']:.1f}%")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()