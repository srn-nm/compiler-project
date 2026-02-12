import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from phase3.analyzer.cfg_builder import ControlFlowGraph, CFGNode, NodeType


class CFGHTMLVisualizer:
    
    def __init__(self):
        self.node_colors = {
            'entry': '#10b981',     
            'exit': '#ef4444',       
            'basic_block': '#3b82f6', 
            'decision': '#f59e0b',   
            'loop': '#8b5cf6',      
            'function': '#ec4899',   
            'return': '#6b7280',     
            'call': '#14b8a6'      
        }
        
        self.node_shapes = {
            'entry': 'ellipse',
            'exit': 'ellipse',
            'basic_block': 'rectangle',
            'decision': 'diamond',
            'loop': 'ellipse',
            'function': 'roundrectangle',
            'return': 'parallelogram',
            'call': 'rectangle'
        }
    
    def generate_cfg_html(self, cfg: ControlFlowGraph, 
                         filename: str = "code.py",
                         output_path: str = "cfg_visualization.html") -> str:
        
        elements = []
        
        for node_id, node in cfg.nodes.items():
            elements.append({
                'data': {
                    'id': str(node_id),
                    'label': f"{node.type.value}\\n{node.label}",
                    'type': node.type.value,
                    'color': self.node_colors.get(node.type.value, '#94a3b8'),
                    'shape': self.node_shapes.get(node.type.value, 'rectangle'),
                    'line': node.line_start or '',
                    'statements': len(node.statements)
                }
            })
        
        for from_id, to_id, edge_data in cfg.edges:
            elements.append({
                'data': {
                    'id': f"e{from_id}-{to_id}",
                    'source': str(from_id),
                    'target': str(to_id),
                    'label': edge_data.get('label', '')
                }
            })
        
        stats = {
            'node_count': len(cfg.nodes),
            'edge_count': len(cfg.edges),
            'cyclomatic': cfg.get_cyclomatic_complexity(),
            'controls': cfg.get_control_structures_count()
        }
        
        html_content = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>نمایش گراف جریان کنترل (CFG) - {filename}</title>
    
    <!-- Cytoscape.js -->
    <script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
    
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Vazirmatn', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 30px 20px;
            color: #e2e8f0;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(10px);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        h1 {{
            font-size: 2em;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            border: 1px solid #334155;
        }}
        
        .file-info {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .file-badge {{
            background: #2563eb;
            color: white;
            padding: 8px 20px;
            border-radius: 50px;
            font-size: 14px;
            font-weight: 500;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 16px;
            border: 1px solid #334155;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 36px;
            font-weight: 700;
            color: #60a5fa;
            line-height: 1;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 14px;
            color: #94a3b8;
        }}
        
        #cy {{
            width: 100%;
            height: 600px;
            background: #0f172a;
            border-radius: 12px;
            border: 1px solid #334155;
        }}
        
        .controls {{
            display: flex;
            gap: 15px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        
        .btn {{
            background: #1e293b;
            border: 1px solid #334155;
            color: #e2e8f0;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-family: 'Vazirmatn', sans-serif;
            font-size: 14px;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .btn:hover {{
            background: #2d3b4f;
            border-color: #60a5fa;
        }}
        
        .btn-primary {{
            background: #2563eb;
            border-color: #3b82f6;
        }}
        
        .btn-primary:hover {{
            background: #1d4ed8;
        }}
        
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            padding: 20px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            margin-top: 30px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: #94a3b8;
        }}
        
        .color-dot {{
            width: 12px;
            height: 12px;
            border-radius: 4px;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #334155;
            color: #94a3b8;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1> گراف جریان کنترل (CFG)</h1>
        
        <div class="header">
            <div class="file-info">
                <span class="file-badge"> {filename}</span>
                <span style="color: #94a3b8;">پیچیدگی سیکلوماتیک: {stats['cyclomatic']}</span>
            </div>
            <div style="color: #60a5fa;">
                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats['node_count']}</div>
                <div class="stat-label">گره</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['edge_count']}</div>
                <div class="stat-label">یال</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['controls'].get('decisions', 0)}</div>
                <div class="stat-label">شرط</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['controls'].get('loops', 0)}</div>
                <div class="stat-label">حلقه</div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="cy.layout({{ name: 'cose', animate: true }}).run()">
                <span></span> مرتب‌سازی خودکار
            </button>
            <button class="btn" onclick="cy.layout({{ name: 'circle', animate: true }}).run()">
                <span></span> دایره‌ای
            </button>
            <button class="btn" onclick="cy.layout({{ name: 'grid', animate: true }}).run()">
                <span></span> شبکه‌ای
            </button>
            <button class="btn" onclick="cy.fit()">
                <span></span> نمایش کامل
            </button>
            <button class="btn btn-primary" onclick="cy.zoom(1); cy.center()">
                <span></span> بازنشانی
            </button>
        </div>
        
        <div id="cy"></div>
        
        <div class="legend">
            <div class="legend-item">
                <span class="color-dot" style="background: #10b981;"></span>
                <span>ورودی</span>
            </div>
            <div class="legend-item">
                <span class="color-dot" style="background: #ef4444;"></span>
                <span>خروجی</span>
            </div>
            <div class="legend-item">
                <span class="color-dot" style="background: #3b82f6;"></span>
                <span>بلوک پایه</span>
            </div>
            <div class="legend-item">
                <span class="color-dot" style="background: #f59e0b;"></span>
                <span>شرط</span>
            </div>
            <div class="legend-item">
                <span class="color-dot" style="background: #8b5cf6;"></span>
                <span>حلقه</span>
            </div>
            <div class="legend-item">
                <span class="color-dot" style="background: #ec4899;"></span>
                <span>تابع</span>
            </div>
            <div class="legend-item">
                <span class="color-dot" style="background: #6b7280;"></span>
                <span>بازگشت</span>
            </div>
            <div class="legend-item">
                <span class="color-dot" style="background: #14b8a6;"></span>
                <span>فراخوانی</span>
            </div>
        </div>
        
        <div class="footer">
            <p>تولید شده توسط CFG Visualizer - Phase 3</p>
            <p style="font-size: 12px; margin-top: 8px;">
                🟡 گره‌ها قابل کشیدن هستند • 🔵 بزرگنمایی با اسکرول • 🟢 دابل کلیک روی گره
            </p>
        </div>
    </div>
    
    <script>
        // ============ داده‌های گراف ============
        const elements = {json.dumps(elements)};
        
        // ============ ساختن گراف با Cytoscape ============
        var cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: elements,
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'background-color': 'data(color)',
                        'label': 'data(label)',
                        'color': '#e2e8f0',
                        'font-size': '11px',
                        'font-family': 'JetBrains Mono, Vazirmatn',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'width': '60px',
                        'height': '40px',
                        'shape': 'data(shape)',
                        'border-width': 2,
                        'border-color': '#ffffff',
                        'border-opacity': 0.8
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 2,
                        'line-color': '#475569',
                        'target-arrow-color': '#94a3b8',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'label': 'data(label)',
                        'color': '#94a3b8',
                        'font-size': '10px',
                        'font-family': 'JetBrains Mono',
                        'text-rotation': 'autorotate',
                        'text-margin-y': -10
                    }}
                }},
                {{
                    selector: 'node:selected',
                    style: {{
                        'border-width': 4,
                        'border-color': '#60a5fa',
                        'background-color': 'data(color)',
                        'border-opacity': 1
                    }}
                }}
            ],
            layout: {{
                name: 'cose',
                animate: true,
                animationDuration: 500,
                fit: true,
                padding: 50,
                nodeRepulsion: 10000,
                idealEdgeLength: 100,
                gravity: 0.5
            }},
            panningEnabled: true,
            userPanningEnabled: true,
            zoomingEnabled: true,
            userZoomingEnabled: true,
            boxSelectionEnabled: false,
            selectionType: 'single'
        }});
        
        // ============ نمایش اطلاعات گره با کلیک ============
        cy.on('tap', 'node', function(evt) {{
            var node = evt.target;
            var type = node.data('type');
            var id = node.data('id');
            var label = node.data('label').replace('\\\\n', ' ');
            var statements = node.data('statements') || 0;
            
            alert(` ${{type}}\\nشناسه: ${{id}}\\nبرچسب: ${{label}}\\nتعداد دستورات: ${{statements}}`);
        }});
        
        // ============ نمایش اطلاعات یال با کلیک ============
        cy.on('tap', 'edge', function(evt) {{
            var edge = evt.target;
            var label = edge.data('label');
            alert(` یال\\nبرچسب: ${{label || 'بدون برچسب'}}`);
        }});
        
        // ============ بزرگنمایی با دابل کلیک ============
        cy.on('dblclick', 'node', function(evt) {{
            var node = evt.target;
            cy.animate({{
                center: {{ eles: node }},
                zoom: 2,
                duration: 500
            }});
        }});
        
        console.log(' گراف با', elements.length, 'عنصر ساخته شد');
    </script>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"    HTML: {output_path}")
        return output_path
    
    def generate_comparison_html(self, cfg1: ControlFlowGraph, cfg2: ControlFlowGraph,
                                file1_name: str = "code1.py",
                                file2_name: str = "code2.py",
                                results: Optional[Dict] = None,
                                output_path: str = "cfg_comparison.html") -> str:
        
        elements1 = []
        for node_id, node in cfg1.nodes.items():
            elements1.append({
                'data': {
                    'id': str(node_id),
                    'label': f"{node.type.value}\\n{node.label}",
                    'type': node.type.value,
                    'color': self.node_colors.get(node.type.value, '#94a3b8'),
                    'shape': self.node_shapes.get(node.type.value, 'rectangle'),
                    'line': node.line_start or '',
                    'statements': len(node.statements)
                }
            })
        
        for from_id, to_id, edge_data in cfg1.edges:
            elements1.append({
                'data': {
                    'id': f"e{from_id}-{to_id}",
                    'source': str(from_id),
                    'target': str(to_id),
                    'label': edge_data.get('label', '')
                }
            })
        
        elements2 = []
        for node_id, node in cfg2.nodes.items():
            elements2.append({
                'data': {
                    'id': str(node_id),
                    'label': f"{node.type.value}\\n{node.label}",
                    'type': node.type.value,
                    'color': self.node_colors.get(node.type.value, '#94a3b8'),
                    'shape': self.node_shapes.get(node.type.value, 'rectangle'),
                    'line': node.line_start or '',
                    'statements': len(node.statements)
                }
            })
        
        for from_id, to_id, edge_data in cfg2.edges:
            elements2.append({
                'data': {
                    'id': f"e{from_id}-{to_id}",
                    'source': str(from_id),
                    'target': str(to_id),
                    'label': edge_data.get('label', '')
                }
            })
        
        # آمار
        stats1 = {
            'node_count': len(cfg1.nodes),
            'edge_count': len(cfg1.edges),
            'cyclomatic': cfg1.get_cyclomatic_complexity(),
            'controls': cfg1.get_control_structures_count()
        }
        
        stats2 = {
            'node_count': len(cfg2.nodes),
            'edge_count': len(cfg2.edges),
            'cyclomatic': cfg2.get_cyclomatic_complexity(),
            'controls': cfg2.get_control_structures_count()
        }
        
        # نمره شباهت
        similarity_score = 0
        if results:
            if 'combined_similarity_score' in results:
                similarity_score = results['combined_similarity_score']
            elif 'cfg_similarity_score' in results:
                similarity_score = results['cfg_similarity_score']
        
        html_content = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>مقایسه CFG - {file1_name} vs {file2_name}</title>
    
    <!-- Cytoscape.js -->
    <script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
    
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Vazirmatn', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 30px 20px;
            color: #e2e8f0;
        }}
        
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(10px);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        h1 {{
            font-size: 2em;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .similarity-banner {{
            background: linear-gradient(145deg, #1e293b, #0f172a);
            border: 1px solid #3b82f6;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .similarity-score {{
            font-size: 48px;
            font-weight: 700;
            color: #60a5fa;
        }}
        
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }}
        
        .graph-card {{
            background: #0f172a;
            border-radius: 16px;
            padding: 20px;
            border: 1px solid #334155;
        }}
        
        .graph-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #334155;
        }}
        
        .graph-title {{
            font-size: 18px;
            font-weight: 600;
            color: #60a5fa;
        }}
        
        .cy-container {{
            width: 100%;
            height: 500px;
            background: #0f172a;
            border-radius: 12px;
            border: 1px solid #334155;
        }}
        
        .stats-mini {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
            padding: 15px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
        }}
        
        .stat-mini-item {{
            text-align: center;
        }}
        
        .stat-mini-value {{
            font-size: 20px;
            font-weight: 600;
            color: #60a5fa;
        }}
        
        .stat-mini-label {{
            font-size: 11px;
            color: #94a3b8;
        }}
        
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: center;
            margin: 30px 0 20px;
            padding: 20px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: #94a3b8;
        }}
        
        .color-dot {{
            width: 12px;
            height: 12px;
            border-radius: 4px;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #334155;
            color: #94a3b8;
            font-size: 13px;
        }}
        
        .controls {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        
        .btn {{
            background: #1e293b;
            border: 1px solid #334155;
            color: #e2e8f0;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-family: 'Vazirmatn', sans-serif;
            font-size: 13px;
        }}
        
        .btn:hover {{
            background: #2d3b4f;
            border-color: #60a5fa;
        }}
        
        @media (max-width: 1200px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1> مقایسه گراف جریان کنترل (CFG)</h1>
        
        <div class="similarity-banner">
            <div>
                <span style="color: #94a3b8;">مقایسه:</span>
                <span style="font-weight: 600; margin-right: 10px;">{file1_name}</span>
                <span style="color: #60a5fa;">↔</span>
                <span style="font-weight: 600;">{file2_name}</span>
            </div>
            <div class="similarity-score">
                {similarity_score:.1f}%
            </div>
        </div>
        
        <div class="comparison-grid">
            <div class="graph-card">
                <div class="graph-header">
                    <span class="graph-title"> {file1_name}</span>
                    <span style="color: #94a3b8;">پیچیدگی: {stats1['cyclomatic']}</span>
                </div>
                <div class="controls">
                    <button class="btn" onclick="cy1.layout({{ name: 'cose', animate: true }}).run()">مرتب‌سازی</button>
                    <button class="btn" onclick="cy1.fit()">نمایش کامل</button>
                    <button class="btn" onclick="cy1.zoom(1); cy1.center()">بازنشانی</button>
                </div>
                <div id="cy1" class="cy-container"></div>
                <div class="stats-mini">
                    <div class="stat-mini-item">
                        <div class="stat-mini-value">{stats1['node_count']}</div>
                        <div class="stat-mini-label">گره</div>
                    </div>
                    <div class="stat-mini-item">
                        <div class="stat-mini-value">{stats1['edge_count']}</div>
                        <div class="stat-mini-label">یال</div>
                    </div>
                    <div class="stat-mini-item">
                        <div class="stat-mini-value">{stats1['controls'].get('decisions', 0)}</div>
                        <div class="stat-mini-label">شرط</div>
                    </div>
                </div>
            </div>
            
            <div class="graph-card">
                <div class="graph-header">
                    <span class="graph-title"> {file2_name}</span>
                    <span style="color: #94a3b8;">پیچیدگی: {stats2['cyclomatic']}</span>
                </div>
                <div class="controls">
                    <button class="btn" onclick="cy2.layout({{ name: 'cose', animate: true }}).run()">مرتب‌سازی</button>
                    <button class="btn" onclick="cy2.fit()">نمایش کامل</button>
                    <button class="btn" onclick="cy2.zoom(1); cy2.center()">بازنشانی</button>
                </div>
                <div id="cy2" class="cy-container"></div>
                <div class="stats-mini">
                    <div class="stat-mini-item">
                        <div class="stat-mini-value">{stats2['node_count']}</div>
                        <div class="stat-mini-label">گره</div>
                    </div>
                    <div class="stat-mini-item">
                        <div class="stat-mini-value">{stats2['edge_count']}</div>
                        <div class="stat-mini-label">یال</div>
                    </div>
                    <div class="stat-mini-item">
                        <div class="stat-mini-value">{stats2['controls'].get('decisions', 0)}</div>
                        <div class="stat-mini-label">شرط</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item"><span class="color-dot" style="background: #10b981;"></span>ورودی</div>
            <div class="legend-item"><span class="color-dot" style="background: #ef4444;"></span>خروجی</div>
            <div class="legend-item"><span class="color-dot" style="background: #3b82f6;"></span>بلوک پایه</div>
            <div class="legend-item"><span class="color-dot" style="background: #f59e0b;"></span>شرط</div>
            <div class="legend-item"><span class="color-dot" style="background: #8b5cf6;"></span>حلقه</div>
            <div class="legend-item"><span class="color-dot" style="background: #ec4899;"></span>تابع</div>
            <div class="legend-item"><span class="color-dot" style="background: #6b7280;"></span>بازگشت</div>
        </div>
        
        <div class="footer">
            <p>تولید شده توسط CFG Visualizer - Phase 3</p>
            <p style="font-size: 12px;">گراف‌های تعاملی با Cytoscape.js</p>
        </div>
    </div>
    
    <script>
        // ============ گراف اول ============
        const elements1 = {json.dumps(elements1)};
        var cy1 = cytoscape({{
            container: document.getElementById('cy1'),
            elements: elements1,
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'background-color': 'data(color)',
                        'label': 'data(label)',
                        'color': '#e2e8f0',
                        'font-size': '10px',
                        'font-family': 'JetBrains Mono, Vazirmatn',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'width': '50px',
                        'height': '30px',
                        'shape': 'data(shape)',
                        'border-width': 2,
                        'border-color': '#fff'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 2,
                        'line-color': '#475569',
                        'target-arrow-color': '#94a3b8',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }}
                }}
            ],
            layout: {{ name: 'cose', animate: true, fit: true, padding: 30 }},
            panningEnabled: true,
            zoomingEnabled: true
        }});
        
        // ============ گراف دوم ============
        const elements2 = {json.dumps(elements2)};
        var cy2 = cytoscape({{
            container: document.getElementById('cy2'),
            elements: elements2,
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'background-color': 'data(color)',
                        'label': 'data(label)',
                        'color': '#e2e8f0',
                        'font-size': '10px',
                        'font-family': 'JetBrains Mono, Vazirmatn',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'width': '50px',
                        'height': '30px',
                        'shape': 'data(shape)',
                        'border-width': 2,
                        'border-color': '#fff'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 2,
                        'line-color': '#475569',
                        'target-arrow-color': '#94a3b8',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }}
                }}
            ],
            layout: {{ name: 'cose', animate: true, fit: true, padding: 30 }},
            panningEnabled: true,
            zoomingEnabled: true
        }});
        
        console.log(' گراف‌ها ساخته شدند');
    </script>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"    HTML comparison: {output_path}")
        return output_path


def generate_cfg_html(cfg: ControlFlowGraph,
                     filename: str = "code.py",
                     output_path: str = "cfg_visualization.html") -> str:
    """Generate interactive HTML visualization for CFG"""
    visualizer = CFGHTMLVisualizer()
    return visualizer.generate_cfg_html(cfg, filename, output_path)


def generate_cfg_comparison_html(cfg1: ControlFlowGraph,
                                cfg2: ControlFlowGraph,
                                file1_name: str = "code1.py",
                                file2_name: str = "code2.py",
                                results: Optional[Dict] = None,
                                output_path: str = "cfg_comparison.html") -> str:
    """Generate interactive HTML comparison for two CFGs"""
    visualizer = CFGHTMLVisualizer()
    return visualizer.generate_comparison_html(
        cfg1, cfg2, file1_name, file2_name, results, output_path
    )