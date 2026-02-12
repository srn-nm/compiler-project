# Phase 3: Control Flow Graph Analysis

## Overview
This module implements Control Flow Graph (CFG) based similarity analysis for plagiarism detection in source code.

## Installation

pip install -r requirements.txt

## How to run


اجرای آنالیز CFG ساده:
python -m phase3.main -f1 phase1/tests/test_python/code1.py -f2 phase1/tests/test_python/code2.py

4. اجرای کامل سه فاز (خودکار):
python -m phase3.main --full -f1 phase1/tests/test_python/code1.py -f2 phase1/tests/test_python/code2.py

5. اجرا با نمایش بصری:
python -m phase3.main -f1 phase1/tests/test_python/code1.py -f2 phase1/tests/test_python/code2.py --visualize --dot cfg.dot
 
6. اجرای مثال‌ها:
python -m phase3.example