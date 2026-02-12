from antlr4 import *
from grammers.generated.python.Python3Lexer import Python3Lexer
from collections import Counter
import math
from typing import List, Dict, Any
import html 

class TokenSimilarityAnalyzer:
    def __init__(self, config: Dict[str, Any] = None):
        self.lexer = None
        self.config = config or {}
        
    def tokenize_code(self, code: str) -> List[Dict[str, Any]]:
        input_stream = InputStream(code)
        self.lexer = Python3Lexer(input_stream)
        token_stream = CommonTokenStream(self.lexer)
        token_stream.fill()
        
        tokens = []
        for token in token_stream.tokens:
            if token.type != -1:  # skip EOF
                token_name = self.lexer.symbolicNames[token.type]
                if token.channel != Token.HIDDEN_CHANNEL:
                    tokens.append({
                        'type': token_name,
                        'text': token.text,
                        'line': token.line,
                        'column': token.column,
                        'index': len(tokens)
                    })
        return tokens
    
    def normalize_tokens(self, tokens: List[Dict[str, Any]]) -> List[str]:
        normalized = []
        for token in tokens:
            if token['type'] in ['NAME', 'IDENTIFIER']:
                normalized.append('IDENTIFIER')
            elif token['type'] in ['NUMBER', 'INT', 'FLOAT']:
                normalized.append('NUMBER')
            elif token['type'] in ['STRING', 'STRING_LITERAL']:
                normalized.append('STRING')
            else:
                normalized.append(token['type'])
        return normalized
    
    def calculate_similarity(self, code1: str, code2: str) -> Dict[str, Any]:
        tokens1 = self.tokenize_code(code1)
        tokens2 = self.tokenize_code(code2)
        
        token_types1 = [token['type'] for token in tokens1]
        token_types2 = [token['type'] for token in tokens2]
        
        normalized1 = self.normalize_tokens(tokens1)
        normalized2 = self.normalize_tokens(tokens2)
        
        metrics = {
            'jaccard_similarity': self.jaccard_similarity(token_types1, token_types2),
            'cosine_similarity': self.cosine_similarity(token_types1, token_types2),
            'levenshtein_similarity': self.levenshtein_similarity(token_types1, token_types2),
            'sequence_similarity': self.sequence_similarity(token_types1, token_types2),
            'normalized_jaccard': self.jaccard_similarity(normalized1, normalized2)
        }
        
        weights = {
            'jaccard_similarity': 0.15,
            'cosine_similarity': 0.25,
            'levenshtein_similarity': 0.25,
            'sequence_similarity': 0.20,
            'normalized_jaccard': 0.15
        }
        
        overall_score = sum(metrics[key] * weights.get(key, 0) 
                          for key in weights if key in metrics)
        
        matched_sections = self.find_matching_sections(tokens1, tokens2)
        variable_patterns = self.analyze_variable_patterns(tokens1, tokens2)
        
        freq1 = Counter(token_types1)
        freq2 = Counter(token_types2)
        
        return {
            'overall_similarity': overall_score * 100,
            'metrics': metrics,
            'token_counts': {
                'code1': len(tokens1),
                'code2': len(tokens2),
                'common_types': len(set(token_types1) & set(token_types2)),
                'unique_types1': len(set(token_types1)),
                'unique_types2': len(set(token_types2))
            },
            'matched_sections': matched_sections,
            'variable_patterns': variable_patterns,
            'token_frequencies': {
                'code1': dict(freq1.most_common(10)),
                'code2': dict(freq2.most_common(10))
            },
            'normalized_similarity': metrics['normalized_jaccard'] * 100,
            'code_lengths': {
                'code1_chars': len(code1),
                'code2_chars': len(code2),
                'code1_lines': code1.count('\n') + 1,
                'code2_lines': code2.count('\n') + 1
            },
            'original_code1': code1,
            'original_code2': code2
        }
    
    def jaccard_similarity(self, list1: List[str], list2: List[str]) -> float:
        set1 = set(list1)
        set2 = set(list2)
        if not set1 and not set2:
            return 1.0
        return len(set1 & set2) / len(set1 | set2)
    
    def cosine_similarity(self, list1: List[str], list2: List[str]) -> float:
        vec1 = Counter(list1)
        vec2 = Counter(list2)
        all_tokens = set(vec1.keys()) | set(vec2.keys())
        v1 = [vec1.get(token, 0) for token in all_tokens]
        v2 = [vec2.get(token, 0) for token in all_tokens]
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def levenshtein_similarity(self, list1: List[str], list2: List[str]) -> float:
        len1, len2 = len(list1), len(list2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if list1[i-1] == list2[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
        max_len = max(len1, len2)
        if max_len == 0:
            return 1.0
        return 1 - (dp[len1][len2] / max_len)
    
    def sequence_similarity(self, list1: List[str], list2: List[str]) -> float:
        len1, len2 = len(list1), len(list2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if list1[i-1] == list2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs_length = dp[len1][len2]
        max_len = max(len1, len2)
        if max_len == 0:
            return 1.0
        return lcs_length / max_len
    
    def find_matching_sections(self, tokens1: List[Dict], tokens2: List[Dict], min_length: int = 3) -> List[Dict]:
        matches = []
        text1 = [f"{t['type']}:{t['text']}" for t in tokens1]
        text2 = [f"{t['type']}:{t['text']}" for t in tokens2]
        
        used_indices1 = set()
        used_indices2 = set()
        
        for i in range(len(text1) - min_length + 1):
            if any(i in range(start, end + 1) for start, end in used_indices1):
                continue
                
            for j in range(len(text2) - min_length + 1):
                if any(j in range(start, end + 1) for start, end in used_indices2):
                    continue
                    
                match_length = 0
                while (i + match_length < len(text1) and 
                       j + match_length < len(text2) and 
                       text1[i + match_length] == text2[j + match_length]):
                    match_length += 1
                
                if match_length >= min_length:
                    used_indices1.add((i, i + match_length - 1))
                    used_indices2.add((j, j + match_length - 1))
                    
                    matches.append({
                        'start1': i,
                        'end1': i + match_length - 1,
                        'start2': j,
                        'end2': j + match_length - 1,
                        'length': match_length,
                        'tokens': tokens1[i:i+match_length],
                        'token_texts': [t['text'] for t in tokens1[i:i+match_length]],
                        'token_types': [t['type'] for t in tokens1[i:i+match_length]],
                        'line_numbers': {
                            'start1': tokens1[i]['line'],
                            'end1': tokens1[i + match_length - 1]['line'],
                            'start2': tokens2[j]['line'],
                            'end2': tokens2[j + match_length - 1]['line']
                        }
                    })
                    break
        
        matches.sort(key=lambda x: x['length'], reverse=True)
        return matches
    
    def analyze_variable_patterns(self, tokens1: List[Dict], tokens2: List[Dict]) -> Dict:
        def extract_variables(tokens):
            variables = {}
            for token in tokens:
                if token['type'] in ['NAME', 'IDENTIFIER']:
                    name = token['text']
                    if name not in variables:
                        variables[name] = {'count': 0, 'positions': []}
                    variables[name]['count'] += 1
                    variables[name]['positions'].append((token['line'], token['column']))
            return variables
        
        vars1 = extract_variables(tokens1)
        vars2 = extract_variables(tokens2)
        common_names = set(vars1.keys()) & set(vars2.keys())
        
        return {
            'count1': len(vars1),
            'count2': len(vars2),
            'common_count': len(common_names),
            'common_names': list(common_names)[:10],
            'patterns': {
                'camel_case': self.count_camel_case(list(vars1.keys())) + self.count_camel_case(list(vars2.keys())),
                'snake_case': self.count_snake_case(list(vars1.keys())) + self.count_snake_case(list(vars2.keys()))
            }
        }
    
    def count_camel_case(self, names: List[str]) -> int:
        count = 0
        for name in names:
            if name and name[0].islower() and any(c.isupper() for c in name[1:]):
                count += 1
        return count
    
    def count_snake_case(self, names: List[str]) -> int:
        count = 0
        for name in names:
            if '_' in name and name.islower():
                count += 1
        return count
    
    def compare_multiple_codes(self, codes: List[str]) -> Dict:
        n = len(codes)
        matrix = [[0] * n for _ in range(n)]
        comparisons = []
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    result = self.calculate_similarity(codes[i], codes[j])
                    similarity = result['overall_similarity'] / 100
                    matrix[i][j] = similarity
                    matrix[j][i] = similarity
                    comparisons.append({
                        'file1': i,
                        'file2': j,
                        'similarity': similarity,
                        'details': result
                    })
        
        return {
            'similarity_matrix': matrix,
            'comparisons': comparisons,
            'num_files': n
        }
    
    def generate_html_report(self, result: Dict, file1_name: str = "code1", file2_name: str = "code2") -> str:
        
        code1 = result.get('original_code1', '')
        code2 = result.get('original_code2', '')
        
        highlighted_code1 = self._highlight_code(code1, result['matched_sections'], is_first=True)
        highlighted_code2 = self._highlight_code(code2, result['matched_sections'], is_first=False)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Code Similarity Analysis Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #1a73e8;
                    border-bottom: 3px solid #1a73e8;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                }}
                .score-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 15px;
                    margin: 20px 0;
                    text-align: center;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                }}
                .score-number {{
                    font-size: 64px;
                    font-weight: bold;
                    line-height: 1;
                }}
                .score-label {{
                    font-size: 18px;
                    opacity: 0.9;
                    margin-top: 10px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .metric-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 12px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    border-left: 5px solid #1a73e8;
                    transition: transform 0.2s;
                }}
                .metric-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
                }}
                .metric-name {{
                    font-size: 14px;
                    color: #7f8c8d;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .metric-value {{
                    font-size: 28px;
                    font-weight: bold;
                    color: #2c3e50;
                    margin-top: 5px;
                }}
                .code-comparison {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                    margin: 30px 0;
                }}
                .code-box {{
                    background: #1e1e1e;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                }}
                .code-header {{
                    background: #2d2d2d;
                    color: #e0e0e0;
                    padding: 15px 20px;
                    font-size: 16px;
                    font-weight: bold;
                    border-bottom: 1px solid #404040;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .code-header span {{
                    background: #1a73e8;
                    color: white;
                    padding: 5px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                }}
                .code-content {{
                    padding: 20px;
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                    font-size: 14px;
                    line-height: 1.6;
                    color: #d4d4d4;
                    overflow-x: auto;
                    background: #1e1e1e;
                }}
                .code-line {{
                    display: flex;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }}
                .line-number {{
                    color: #858585;
                    padding-right: 20px;
                    text-align: right;
                    user-select: none;
                    width: 50px;
                    flex-shrink: 0;
                }}
                .line-content {{
                    flex: 1;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }}
                .highlight {{
                    background-color: #ffd700;
                    color: #000000 !important;
                    padding: 2px 4px;
                    border-radius: 4px;
                    font-weight: bold;
                }}
                .match-info {{
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 15px 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                }}
                .match-badge {{
                    display: inline-block;
                    background: #ffc107;
                    color: #000;
                    padding: 5px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: bold;
                    margin-right: 10px;
                }}
                .stats-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 30px 0;
                    background: white;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                }}
                .stats-table th {{
                    background: #1a73e8;
                    color: white;
                    padding: 15px;
                    text-align: left;
                }}
                .stats-table td {{
                    padding: 12px 15px;
                    border-bottom: 1px solid #ecf0f1;
                }}
                .stats-table tr:hover {{
                    background: #f8f9fa;
                }}
                .token-cloud {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin: 20px 0;
                }}
                .token-tag {{
                    background: #e9ecef;
                    padding: 8px 15px;
                    border-radius: 20px;
                    font-size: 13px;
                    color: #495057;
                    border: 1px solid #dee2e6;
                }}
                .token-tag strong {{
                    color: #1a73e8;
                    margin-left: 5px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #ecf0f1;
                    color: #7f8c8d;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Code Similarity Analysis Report - Phase 1</h1>
                
                <div class="score-card">
                    <div class="score-number">{result['overall_similarity']:.1f}%</div>
                    <div class="score-label">Overall Similarity Score</div>
                    <div style="margin-top: 15px; font-size: 16px;">
                        Normalized: {result.get('normalized_similarity', 0):.1f}%
                    </div>
                </div>
                
                <div style="display: flex; justify-content: space-between; margin: 20px 0;">
                    <div style="background: #e8f5e9; padding: 15px; border-radius: 10px; flex: 1; margin-right: 10px;">
                        <strong>File 1:</strong> {file1_name}<br>
                        <small>{result['code_lengths']['code1_lines']} lines • {result['token_counts']['code1']} tokens • {result['token_counts']['unique_types1']} unique types</small>
                    </div>
                    <div style="background: #e8f5e9; padding: 15px; border-radius: 10px; flex: 1; margin-left: 10px;">
                        <strong>File 2:</strong> {file2_name}<br>
                        <small>{result['code_lengths']['code2_lines']} lines • {result['token_counts']['code2']} tokens • {result['token_counts']['unique_types2']} unique types</small>
                    </div>
                </div>
                
                <h2>Similarity Metrics</h2>
                <div class="metrics-grid">
        """
        
        for metric, value in result['metrics'].items():
            if isinstance(value, (int, float)) and metric not in ['normalized_jaccard']:
                metric_name = metric.replace('_', ' ').title()
                html += f"""
                    <div class="metric-card">
                        <div class="metric-name">{metric_name}</div>
                        <div class="metric-value">{value*100:.1f}%</div>
                    </div>
                """
        
        html += f"""
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 30px 0;">
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 12px;">
                        <h3 style="margin-top: 0; color: #1a73e8;">Matching Sections</h3>
                        <p style="font-size: 24px; font-weight: bold; margin: 10px 0;">{len(result['matched_sections'])}</p>
                        <p style="color: #666;">matching code sections found</p>
                    </div>
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 12px;">
                        <h3 style="margin-top: 0; color: #1a73e8;">Variable Analysis</h3>
                        <p style="font-size: 24px; font-weight: bold; margin: 10px 0;">{result['variable_patterns']['common_count']}</p>
                        <p style="color: #666;">common variable names</p>
                    </div>
                </div>
                
                <h2>Visual Code Comparison</h2>
                <div class="code-comparison">
                    <div class="code-box">
                        <div class="code-header">
                            {file1_name}
                            <span>{result['token_counts']['code1']} tokens</span>
                        </div>
                        <div class="code-content">
                            {highlighted_code1}
                        </div>
                    </div>
                    <div class="code-box">
                        <div class="code-header">
                            {file2_name}
                            <span>{result['token_counts']['code2']} tokens</span>
                        </div>
                        <div class="code-content">
                            {highlighted_code2}
                        </div>
                    </div>
                </div>
                
                <div class="match-info">
                    <h3 style="margin-top: 0; color: #856404;">Matching Sections Details</h3>
        """
        
        for i, match in enumerate(result['matched_sections'][:10], 1):
            html += f"""
                    <div style="margin: 15px 0; padding: 15px; background: white; border-radius: 8px; border: 1px solid #ffeeba;">
                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                            <span class="match-badge">Match {i}</span>
                            <span style="color: #666;">{match['length']} tokens • Lines {match['line_numbers']['start1']}-{match['line_numbers']['end1']} ↔ Lines {match['line_numbers']['start2']}-{match['line_numbers']['end2']}</span>
                        </div>
                        <code style="background: #f8f9fa; padding: 10px; border-radius: 5px; display: block; color: #2c3e50;">
                            {' '.join(match['token_texts'][:15])}{'...' if len(match['token_texts']) > 15 else ''}
                        </code>
                    </div>
            """
        
        html += f"""
                </div>
                
                <h2>Token Statistics</h2>
                <table class="stats-table">
                    <tr>
                        <th>Metric</th>
                        <th>File 1</th>
                        <th>File 2</th>
                        <th>Common</th>
                    </tr>
                    <tr>
                        <td><strong>Token Count</strong></td>
                        <td>{result['token_counts']['code1']}</td>
                        <td>{result['token_counts']['code2']}</td>
                        <td>{result['token_counts']['common_types']}</td>
                    </tr>
                    <tr>
                        <td><strong>Unique Types</strong></td>
                        <td>{result['token_counts']['unique_types1']}</td>
                        <td>{result['token_counts']['unique_types2']}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td><strong>Lines of Code</strong></td>
                        <td>{result['code_lengths']['code1_lines']}</td>
                        <td>{result['code_lengths']['code2_lines']}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td><strong>Characters</strong></td>
                        <td>{result['code_lengths']['code1_chars']}</td>
                        <td>{result['code_lengths']['code2_chars']}</td>
                        <td>-</td>
                    </tr>
                </table>
                
                <h2>Most Common Tokens</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                    <div>
                        <h4 style="color: #1a73e8;">File 1: {file1_name}</h4>
                        <div class="token-cloud">
        """
        
        for token, count in list(result['token_frequencies']['code1'].items())[:15]:
            html += f'<span class="token-tag">{token} <strong>{count}</strong></span>'
        
        html += f"""
                        </div>
                    </div>
                    <div>
                        <h4 style="color: #1a73e8;">File 2: {file2_name}</h4>
                        <div class="token-cloud">
        """
        
        for token, count in list(result['token_frequencies']['code2'].items())[:15]:
            html += f'<span class="token-tag">{token} <strong>{count}</strong></span>'
        
        html += f"""
                        </div>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Generated by TokenSimilarityAnalyzer - Phase 1</p>
                    <p style="font-size: 12px;">
                        Highlighted sections show identical token sequences<br>
                        Analysis based on lexical tokens, ignoring whitespace and comments
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _highlight_code(self, code: str, matches: List[Dict], is_first: bool = True) -> str:
        lines = code.split('\n')
        highlighted_lines = []
        highlight_lines = set()
        line_ranges = []
        
        for match in matches:
            if is_first:
                start_line = match['line_numbers']['start1']
                end_line = match['line_numbers']['end1']
            else:
                start_line = match['line_numbers']['start2']
                end_line = match['line_numbers']['end2']
            
            line_ranges.append((start_line, end_line))
            for line_num in range(start_line, end_line + 1):
                highlight_lines.add(line_num)
        
        for i, line in enumerate(lines, 1):
            line_escaped = html.escape(line)
            line_class = 'highlight-line' if i in highlight_lines else ''
            
            range_info = ''
            for start, end in line_ranges:
                if start <= i <= end:
                    range_info = f' data-match="lines-{start}-{end}"'
                    break
            
            highlighted_lines.append(
                f'<div class="code-line">'
                f'<span class="line-number">{i}</span>'
                f'<span class="line-content {"highlight" if i in highlight_lines else ""}"{range_info}>'
                f'{line_escaped or " "}'
                f'</span>'
                f'</div>'
            )
        return '\n'.join(highlighted_lines)