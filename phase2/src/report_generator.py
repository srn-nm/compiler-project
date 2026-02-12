from typing import Dict, Any, Optional
from datetime import datetime
from anytree import Node, RenderTree
import html


def ast_to_anytree(ast_dict: Optional[Dict], parent: Optional[Node] = None) -> Optional[Node]:
    if not ast_dict:
        return None
    
    node_type = ast_dict.get('type', 'Unknown')
    node_value = ast_dict.get('value')
    node_line = ast_dict.get('line')
    
    if node_value and node_value not in ['FUNC', 'VAR', 'CLASS', 'STR', 'INT', 'FLOAT']:
        name = f"{node_type}: {node_value}"
    else:
        name = f"{node_type}"
    
    if node_line:
        name += f" (L{node_line})"
    
    node = Node(name, parent=parent)
    
    for child in ast_dict.get('children', []):
        ast_to_anytree(child, node)
    
    return node


def render_ast_with_anytree(ast_dict: Optional[Dict]) -> str:
    if not ast_dict:
        return "<p style='color: #94a3b8;'> داده AST موجود نیست</p>"
    
    root = ast_to_anytree(ast_dict)
    if not root:
        return "<p style='color: #94a3b8;'> خطا در ساخت درخت</p>"
    
    lines = []
    for pre, _, node in RenderTree(root):
        prefix = pre.replace(" ", "&nbsp;").replace("├──", "├─").replace("└──", "└─")
        lines.append(f"{prefix}{node.name}")
    
    return "<br>".join(lines)


class Phase2HTMLReportGenerator:    
    def __init__(self):
        pass
    
    def generate_report(self, results: Dict[str, Any], 
                       file1_name: str = "code1.py",
                       file2_name: str = "code2.py",
                       output_path: str = "phase2_report.html") -> str:
        
        combined_score = results.get('combined_similarity_score', results.get('ast_similarity_score', 0))
        token_score = results.get('token_similarity_score', 0)
        ast_score = results.get('ast_similarity_score', 0)
        
        is_plagiarism = results.get('is_plagiarism_suspected', False)
        threshold = results.get('threshold_used', 0.65) * 100
        decision = results.get('final_decision', 'PLAGIARISM_SUSPECTED' if is_plagiarism else 'CLEAN')
        
        ast_stats = results.get('phase2_details', {}).get('ast_statistics', {})
        stats1 = ast_stats.get('code1', {})
        stats2 = ast_stats.get('code2', {})
        
        ast_metrics = results.get('phase2_details', {}).get('ast_similarity_metrics', 
                      results.get('ast_similarity_metrics', {}))
        
        matched_nodes = results.get('phase2_details', {}).get('matched_nodes_count',
                            results.get('matched_nodes_count', 0))
        matched_samples = results.get('phase2_details', {}).get('matched_nodes_sample',
                            results.get('matched_nodes_sample', []))
        
        phase1_details = results.get('phase1_details', {})
        phase1_score = phase1_details.get('overall_similarity', token_score)
        phase1_matches = phase1_details.get('matched_sections_count', 0)
        
        ast1_dict = results.get('ast1_dict')
        ast2_dict = results.get('ast2_dict')
        
        tree1_html = render_ast_with_anytree(ast1_dict)
        tree2_html = render_ast_with_anytree(ast2_dict)
        
        html_content = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>گزارش تحلیل فاز ۲ - شباهت ساختاری (AST)</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Vazirmatn', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f172a;
            padding: 30px 20px;
            color: #e2e8f0;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: #1e293b;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid #334155;
        }}
        
        h1 {{
            font-size: 2em;
            font-weight: 700;
            color: #60a5fa;
            margin-bottom: 20px;
            text-align: center;
            padding-bottom: 15px;
            border-bottom: 2px solid #334155;
        }}
        
        h2 {{
            color: #94a3b8;
            margin: 30px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #334155;
            font-weight: 600;
        }}
        
        h3 {{
            color: #e2e8f0;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        
        .score-card {{
            background: #0f172a;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid #3b82f6;
        }}
        
        .score-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            text-align: center;
        }}
        
        .score-item {{
            padding: 20px;
            background: #1e293b;
            border-radius: 8px;
            border: 1px solid #334155;
        }}
        
        .score-label {{
            font-size: 14px;
            color: #94a3b8;
            margin-bottom: 10px;
        }}
        
        .score-value {{
            font-size: 48px;
            font-weight: 700;
            color: #60a5fa;
            line-height: 1;
            margin-bottom: 5px;
        }}
        
        .score-unit {{
            font-size: 16px;
            color: #94a3b8;
        }}
        
        .decision-badge {{
            display: inline-block;
            padding: 10px 24px;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            margin-top: 20px;
        }}
        
        .badge-plagiarism {{
            background: #dc2626;
            color: white;
        }}
        
        .badge-clean {{
            background: #059669;
            color: white;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: #0f172a;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #334155;
        }}
        
        .metric-name {{
            font-size: 13px;
            color: #94a3b8;
            margin-bottom: 10px;
        }}
        
        .metric-value {{
            font-size: 24px;
            font-weight: 700;
            color: #60a5fa;
        }}
        
        .stats-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 30px 0;
        }}
        
        .stat-box {{
            background: #0f172a;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #334155;
        }}
        
        .stat-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #334155;
        }}
        
        .stat-header h3 {{
            color: #e2e8f0;
            font-size: 18px;
        }}
        
        .ast-tree {{
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            font-size: 13px;
            line-height: 1.8;
            color: #cbd5e1;
            background: #0f172a;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid #334155;
            max-height: 500px;
            overflow-y: auto;
            direction: ltr;
            text-align: left;
        }}
        
        .node-type {{
            color: #60a5fa;
            font-weight: 600;
        }}
        
        .matched-nodes {{
            margin: 30px 0;
        }}
        
        .match-table {{
            width: 100%;
            border-collapse: collapse;
            background: #0f172a;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .match-table th {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 12px;
            font-weight: 600;
            text-align: right;
        }}
        
        .match-table td {{
            padding: 12px;
            border-bottom: 1px solid #334155;
            color: #cbd5e1;
        }}
        
        .match-table tr:hover {{
            background: #1e293b;
        }}
        
        .similarity-bar {{
            width: 100%;
            height: 8px;
            background: #334155;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .similarity-fill {{
            height: 100%;
            background: #60a5fa;
            border-radius: 4px;
        }}
        
        .phase-integration {{
            background: #1e1b4b;
            border-radius: 8px;
            padding: 25px;
            margin: 30px 0;
            border: 1px solid #818cf8;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 30px;
            border-top: 1px solid #334155;
            color: #94a3b8;
        }}
        
        .tree-title {{
            color: #60a5fa;
            font-weight: 600;
            margin-bottom: 15px;
            font-size: 16px;
        }}
        
        @media (max-width: 1200px) {{
            .score-grid,
            .metrics-grid,
            .stats-container {{
                grid-template-columns: 1fr;
            }}
            
            .container {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1> گزارش تحلیل فاز ۲ - شباهت ساختاری (AST)</h1>
        
        <!-- Score Card -->
        <div class="score-card">
            <div class="score-grid">
                <div class="score-item">
                    <div class="score-label">شباهت ترکیبی</div>
                    <div class="score-value">{combined_score:.1f}</div>
                    <div class="score-unit">%</div>
                </div>
                <div class="score-item">
                    <div class="score-label">فاز ۱ - توکن</div>
                    <div class="score-value">{token_score:.1f}</div>
                    <div class="score-unit">%</div>
                </div>
                <div class="score-item">
                    <div class="score-label">فاز ۲ - AST</div>
                    <div class="score-value">{ast_score:.1f}</div>
                    <div class="score-unit">%</div>
                </div>
            </div>
            
            <div style="text-align: center;">
                <span class="decision-badge {'badge-plagiarism' if is_plagiarism else 'badge-clean'}">
                    {' سرقت ادبی مشکوک' if is_plagiarism else ' بدون سرقت'}
                </span>
                <div style="margin-top: 15px; color: #94a3b8;">
                    آستانه تشخیص: {threshold:.0f}% | تصمیم نهایی: {decision}
                </div>
            </div>
        </div>
        
        <!-- AST Metrics -->
        <h2> معیارهای شباهت ساختاری</h2>
        <div class="metrics-grid">
"""
        
        metric_names = {
            'structural_similarity': 'شباهت ساختاری',
            'node_type_similarity': 'شباهت نوع گره',
            'subtree_similarity': 'شباهت زیردرخت',
            'depth_similarity': 'شباهت عمق'
        }
        
        for metric, value in ast_metrics.items():
            if metric in metric_names and isinstance(value, (int, float)):
                html_content += f"""
            <div class="metric-card">
                <div class="metric-name">{metric_names[metric]}</div>
                <div class="metric-value">{value*100:.1f}%</div>
            </div>
                """
        
        html_content += f"""
        </div>
        
        <!-- AST Statistics -->
        <h2>آمار درخت نحوی</h2>
        <div class="stats-container">
            <div class="stat-box">
                <div class="stat-header">
                    <h3>{file1_name}</h3>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>تعداد گره‌ها:</div>
                    <div style="color: #60a5fa; font-weight: 600;">{stats1.get('total_nodes', 0)}</div>
                    <div>عمق درخت:</div>
                    <div style="color: #60a5fa;">{stats1.get('max_depth', 0)}</div>
                    <div>انواع گره:</div>
                    <div style="color: #60a5fa;">{stats1.get('node_types_count', 0)}</div>
                </div>
            </div>
            <div class="stat-box">
                <div class="stat-header">
                    <h3>{file2_name}</h3>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>تعداد گره‌ها:</div>
                    <div style="color: #60a5fa; font-weight: 600;">{stats2.get('total_nodes', 0)}</div>
                    <div>عمق درخت:</div>
                    <div style="color: #60a5fa;">{stats2.get('max_depth', 0)}</div>
                    <div>انواع گره:</div>
                    <div style="color: #60a5fa;">{stats2.get('node_types_count', 0)}</div>
                </div>
            </div>
        </div>
        
        <!-- Matched Nodes -->
        <h2>1گره‌های مشابه</h2>
        <div class="matched-nodes">
            <div style="background: #0f172a; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <div style="font-size: 18px; font-weight: 600; color: #60a5fa; margin-bottom: 10px;">
                    {matched_nodes} گره مشابه یافت شد
                </div>
                <div class="similarity-bar">
                    <div class="similarity-fill" style="width: {ast_score}%;"></div>
                </div>
            </div>
"""
        
        if matched_samples and len(matched_samples) > 0:
            html_content += f"""
            <table class="match-table">
                <thead>
                    <tr>
                        <th>کد اول</th>
                        <th>کد دوم</th>
                        <th>شباهت</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for node in matched_samples[:10]:
                similarity = node.get('similarity', 0) * 100
                node1_type = node.get('node1_type', 'Unknown')
                node2_type = node.get('node2_type', 'Unknown')
                node1_line = node.get('node1_line', '?')
                node2_line = node.get('node2_line', '?')
                
                html_content += f"""
                    <tr>
                        <td>{node1_type} (خط {node1_line})</td>
                        <td>{node2_type} (خط {node2_line})</td>
                        <td style="color: {'#10b981' if similarity > 80 else '#fbbf24'}; font-weight: 600;">
                            {similarity:.1f}%
                        </td>
                    </tr>
                """
            
            html_content += """
                </tbody>
            </table>
            """
        else:
            html_content += """
            <div style="background: #0f172a; padding: 20px; border-radius: 8px; text-align: center; color: #94a3b8;">
                هیچ گره مشابهی یافت نشد
            </div>
            """
        
        html_content += f"""
        </div>
        
        <!-- Visual AST Trees with anytree -->
        <h2> نمایش درختی AST (anytree)</h2>
        <div class="stats-container">
            <div class="stat-box">
                <div class="tree-title">{file1_name}</div>
                <div class="ast-tree">
                    {tree1_html}
                </div>
            </div>
            <div class="stat-box">
                <div class="tree-title">{file2_name}</div>
                <div class="ast-tree">
                    {tree2_html}
                </div>
            </div>
        </div>
        
        <!-- Phase 1 Integration -->
        <div class="phase-integration">
            <h3 style="color: #818cf8; margin-bottom: 20px;"> یکپارچه‌سازی با فاز ۱</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                <div>
                    <div style="color: #94a3b8;">شباهت توکن</div>
                    <div style="font-size: 24px; font-weight: 700; color: #818cf8;">{phase1_score:.1f}%</div>
                </div>
                <div>
                    <div style="color: #94a3b8;">بخش‌های مشابه</div>
                    <div style="font-size: 24px; font-weight: 700; color: #818cf8;">{phase1_matches}</div>
                </div>
                <div>
                    <div style="color: #94a3b8;">شباهت ترکیبی</div>
                    <div style="font-size: 24px; font-weight: 700; color: #818cf8;">{combined_score:.1f}%</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>تولید شده توسط Phase 2 AST Analyzer - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="font-size: 12px; margin-top: 10px; color: #64748b;">
                 نمایش درختی با anytree • گره‌های آبی: نوع گره • اعداد: شماره خط
            </p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report saved: {output_path}")
        return output_path


def generate_phase2_html_report(results: Dict[str, Any],
                               file1_name: str = "code1.py",
                               file2_name: str = "code2.py",
                               output_path: str = "phase2_report.html") -> str:
    generator = Phase2HTMLReportGenerator()
    return generator.generate_report(results, file1_name, file2_name, output_path)


def generate_integrated_html_report(phase1_results: Dict[str, Any],
                                   phase2_results: Dict[str, Any],
                                   file1_name: str = "code1.py",
                                   file2_name: str = "code2.py",
                                   output_path: str = "integrated_report.html") -> str:
    
    combined_results = {
        **phase2_results,
        'phase1_details': {
            'overall_similarity': phase1_results.get('overall_similarity', 0),
            'token_metrics': phase1_results.get('similarity_metrics', {}),
            'matched_sections_count': len(phase1_results.get('matched_sections', [])),
            'token_counts': phase1_results.get('token_counts', {})
        },
        'combined_similarity_score': phase2_results.get('combined_similarity_score', 
            (phase1_results.get('overall_similarity', 0) * 0.3 + 
             phase2_results.get('ast_similarity_score', 0) * 0.7)),
        'token_similarity_score': phase1_results.get('overall_similarity', 0),
        'is_integrated': True
    }
    
    generator = Phase2HTMLReportGenerator()
    return generator.generate_report(combined_results, file1_name, file2_name, output_path)