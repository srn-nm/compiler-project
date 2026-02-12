"""
Bridge between Phase 1 and Phase 2
Assuming Phase 1 is in the phase1 directory
"""


import os
import subprocess
import json
import tempfile
from typing import Dict, Optional, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # ریشه پروژه

def run_phase1_simple(code1: str, code2: str, config_path: str = None) -> Optional[Dict]:

    phase1_dir = find_phase1_directory()
    if not phase1_dir:
        print(" Phase 1 directory not found")
        return None

    phase1_main = os.path.join(phase1_dir, 'main.py')
    if not os.path.exists(phase1_main):
        print(f" Phase 1 main.py file not found at {phase1_main}")
        return None

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f1:
        f1.write(code1)
        temp_file1 = f1.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f2:
        f2.write(code2)
        temp_file2 = f2.name

    output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False).name

    try:
        cmd = [sys.executable, phase1_main, temp_file1, temp_file2, '--output', output_file]

        if config_path:
            cmd.extend(['--config', config_path])

        print(f"Running Phase 1...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=phase1_dir)

        if result.returncode == 0:
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print("Phase 1 executed successfully")
                return results
            else:
                print(f"Phase 1 output file not created: {output_file}")
        else:
            print(f"Error executing Phase 1:")
            print(f"   stderr: {result.stderr[:200]}")

    except Exception as e:
        print(f"Exception executing Phase 1: {e}")

    finally:
        for f in [temp_file1, temp_file2, output_file]:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except:
                pass

    return None


def find_phase1_directory() -> Optional[str]:
    """Find Phase 1 directory"""
    # Search in possible paths
    possible_paths = [
        'phase1',
        '../phase1',
        '../../phase1',
        os.path.join(os.path.dirname(__file__), '..', 'phase1'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'phase1'),
    ]

    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            main_py = os.path.join(abs_path, 'main.py')
            analyzer_py = os.path.join(abs_path, 'src', 'token_similarity_analyzer.py')
            if os.path.exists(main_py) or os.path.exists(analyzer_py):
                return abs_path

    return None


def load_phase1_results(file_path: str) -> Dict:
    """Load Phase 1 results from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_phase1_metrics(phase1_results: Dict) -> Dict:
    """Extract important metrics from Phase 1 results"""
    return {
        'overall_similarity': phase1_results.get('overall_similarity', 0),
        'token_metrics': phase1_results.get('similarity_metrics', {}),
        'token_counts': phase1_results.get('token_counts', {}),
        'matched_sections': phase1_results.get('matched_sections', []),
        'common_functions': phase1_results.get('common_functions', [])
    }


def create_ast_analysis_input(phase1_results: Dict, code1: str, code2: str) -> Dict:

    similar_regions = []
    for match in phase1_results.get('matched_sections', []):
        similar_regions.append({
            'type': 'token_match',
            'similarity': match.get('similarity', 0),
            'length': match.get('length', 0)
        })

    similar_functions = []
    for func in phase1_results.get('common_functions', []):
        similar_functions.append({
            'name': func.get('name', ''),
            'similarity': func.get('similarity', 0)
        })

    return {
        'similar_regions': similar_regions,
        'similar_functions': similar_functions,
        'overall_token_similarity': phase1_results.get('overall_similarity', 0) / 100,
        'focus_areas': similar_regions[:5]
    }