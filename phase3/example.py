"""
Example usage of phase 3
"""
from analyzer.cfg_analyzer import Phase3CFGSimilarity, CFGAnalyzer
from visualizer.cfg_visualizer import visualize_cfg, generate_cfg_report

def example1():
    """Simple example: Analyzing two simple functions"""
    print("Example 1: Analyzing two sum calculation functions")
    print("=" * 60)

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

    print(f"CFG Similarity Score: {results['cfg_similarity_score']:.2f}%")
    print(f"Detection: {'Similar' if results['is_plagiarism_suspected'] else 'Not Similar'}")

    return results

def example2():
    """More complex example: Analyzing codes with different control structures"""
    print("\nExample 2: Analyzing codes with complex structures")
    print("=" * 60)

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

    report = generate_cfg_report(results)
    print(report)

    return results


def example3():
    """Example using previous phases"""
    print("\nExample 3: Integrated analysis with phases 1 and 2")
    print("=" * 60)

    code1 = """
def factorial_iterative(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result
"""

    code2 = """
def fact_recursive(x):
    if x <= 1:
        return 1
    return x * fact_recursive(x-1)
"""

    phase1_results = {
        'overall_similarity': 45.5,
        'token_counts': {'code1': 25, 'code2': 20, 'common': 10},
        'matched_sections': []
    }

    phase2_results = {
        'ast_similarity_score': 38.2,
        'ast_statistics': {
            'code1': {'total_nodes': 15},
            'code2': {'total_nodes': 12}
        },
        'is_plagiarism_suspected': False
    }

    analyzer = Phase3CFGSimilarity()
    results = analyzer.analyze_code_pair(
        code1, code2, phase1_results, phase2_results
    )

    print(f"Combined Score: {results.get('combined_similarity_score', 0):.2f}%")
    print(f"CFG Score: {results.get('cfg_similarity_score', 0):.2f}%")
    print(f"Final Decision: {results.get('final_decision', 'Unknown')}")

    return results


def example4_visualization():
    """CFG Visualization Example"""
    print("\nExample 4: CFG Visualization")
    print("=" * 60)

    code = """
def find_max(numbers):
    if not numbers:
        return None

    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num

    return max_num
"""

    analyzer = CFGAnalyzer()
    ast1, _ = analyzer._get_asts_from_phase2(code, "")
    cfg = analyzer.build_cfg_from_ast(ast1)

    visualization = visualize_cfg(cfg, max_nodes=20)
    print(visualization)
    from visualizer.cfg_visualizer import visualize_cfg, generate_cfg_report
    generate_cfg_report(cfg, "example_cfg.dot")
    print("DOT file saved at example_cfg.dot")


def main():
    """Main function"""
    print("Phase 3 Project Usage Examples")
    print("=" * 60)

    results1 = example1()
    results2 = example2()
    results3 = example3()
    example4_visualization()

    # Summary
    print("\n" + "=" * 60)
    print("Results Summary:")
    print(
        f"Example 1 (simple functions): {results1['cfg_similarity_score']:.1f}% - {results1['is_plagiarism_suspected']}")
    print(
        f"Example 2 (complex structure): {results2['cfg_similarity_score']:.1f}% - {results2['is_plagiarism_suspected']}")

    if 'combined_similarity_score' in results3:
        print(
            f"Example 3 (integrated): {results3['combined_similarity_score']:.1f}% - {results3['is_plagiarism_suspected']}")
    else:
        print(
            f"Example 3 (integrated): {results3['cfg_similarity_score']:.1f}% - {results3['is_plagiarism_suspected']}")


if __name__ == "__main__":
    main()