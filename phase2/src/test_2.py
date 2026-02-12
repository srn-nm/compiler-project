"""
Simple tests for Phase 2
"""

import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def test_basic_imports():
    """Test basic imports"""
    print("🔧 Testing module imports...")

    try:
        from analyzer import ASTSimilarityAnalyzer, Phase2ASTSimilarity
        print("analyzer imported successfully")

        from utils import preprocess_code, extract_code_blocks
        print("utils imported successfully")

        from similarity import calculate_ast_similarity
        print("similarity imported successfully")

        from visualizer import generate_phase2_report
        print("visualizer imported successfully")

        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False


def test_simple_analysis():
    """Test simple analysis"""
    print("\nTesting simple code analysis...")

    try:
        from analyzer import Phase2ASTSimilarity

        code1 = "def add(a, b): return a + b"
        code2 = "def sum(x, y): return x + y"

        analyzer = Phase2ASTSimilarity()
        results = analyzer.analyze_code_pair(code1, code2)

        if 'ast_similarity_score' in results:
            print(f"Analysis successful - Score: {results['ast_similarity_score']:.1f}%")
            return True
        else:
            print("Analysis failed")
            return False

    except Exception as e:
        print(f"Analysis error: {e}")
        return False


def test_visualization():
    """Test visualization"""
    print("\n🎨 Testing report generation...")

    try:
        from visualizer import generate_phase2_report

        sample_results = {
            'ast_similarity_score': 85.5,
            'is_plagiarism_suspected': True,
            'threshold_used': 0.65
        }

        report = generate_phase2_report(sample_results)

        if len(report) > 100 and "PHASE 2 ANALYSIS REPORT" in report:
            print("\nReport generated successfully")
            return True
        else:
            print("Report generation failed")
            return False

    except Exception as e:
        print(f"Report generation error: {e}")
        return False


def test_integration():
    """Test integration"""
    print("\n🔗 Testing integration...")

    try:
        from similarity import integrate_with_phase1

        phase1_results = {
            'overall_similarity': 75.0,
            'token_counts': {'code1': 10, 'code2': 10}
        }

        code1 = "x = 10"
        code2 = "y = 20"

        results = integrate_with_phase1(phase1_results, code1, code2)

        if 'combined_similarity_score' in results:
            print(f"Integration successful - Combined score: {results['combined_similarity_score']:.1f}%")
            return True
        else:
            print("Integration failed")
            return False

    except Exception as e:
        print(f"Integration error: {e}")
        return False


def run_all_tests():
    """Run all simple tests"""
    print("=" * 60)
    print("Running essential Phase 2 tests")
    print("=" * 60)

    tests = [
        ("Module imports", test_basic_imports),
        ("Code analysis", test_simple_analysis),
        ("Report generation", test_visualization),
        ("Integration", test_integration)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n-->Running: {test_name}")
        if test_func():
            passed += 1
            print(f"Success")
        else:
            print(f"Failed")

    # Results
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed successfully!")
        return True
    else:
        print(f"{total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)