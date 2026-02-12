#!/usr/bin/env python3
"""
اجرای کامل پروژه - یکپارچه‌سازی فازهای ۱، ۲ و ۳
"""

import sys
import json
import argparse
from pathlib import Path

# اضافه کردن ریشه پروژه به PATH برای import صحیح
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ========== ایمپورت فاز ۱ ==========
try:
    from phase1.src.token_similarity_analyzer import TokenSimilarityAnalyzer
    PHASE1_OK = True
except ImportError as e:
    print("⚠️ فاز ۱ یافت نشد:", e)
    PHASE1_OK = False

# ========== ایمپورت فاز ۲ ==========
try:
    from phase2.src.analyzer import Phase2ASTSimilarity
    PHASE2_OK = True
except ImportError as e:
    print("⚠️ فاز ۲ یافت نشد:", e)
    PHASE2_OK = False

# ========== ایمپورت فاز ۳ ==========
try:
    from phase3.analyzer.cfg_analyzer import Phase3CFGSimilarity
    PHASE3_OK = True
except ImportError as e:
    print("⚠️ فاز ۳ یافت نشد:", e)
    PHASE3_OK = False


def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description='تشخیص سرقت ادبی در کد - سه فاز کامل')
    parser.add_argument('file1', help='فایل کد اول')
    parser.add_argument('file2', help='فایل کد دوم')
    parser.add_argument('--lang', '-l', default='python', help='زبان برنامه‌نویسی (پیشفرض: python)')
    parser.add_argument('--output', '-o', default='final_report.json', help='فایل خروجی JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='نمایش جزئیات')
    args = parser.parse_args()

    code1 = read_file(args.code1)
    code2 = read_file(args.code2)

    print("=" * 70)
    print("🧪  تحلیل سرقت ادبی - اجرای سه فاز")
    print("=" * 70)

    # ---------- فاز ۱ ----------
    phase1_res = None
    if PHASE1_OK:
        print("\n[فاز ۱] تحلیل توکن ...")
        analyzer1 = TokenSimilarityAnalyzer()
        phase1_res = analyzer1.calculate_similarity(code1, code2)
        print(f"   ✅ شباهت توکن: {phase1_res.get('overall_similarity', 0):.2f}%")

    # ---------- فاز ۲ ----------
    phase2_res = None
    ast1_dict = None
    ast2_dict = None
    if PHASE2_OK:
        print("\n[فاز ۲] تحلیل درخت نحوی (AST) ...")
        analyzer2 = Phase2ASTSimilarity()
        phase2_res = analyzer2.analyze_code_pair(code1, code2, args.lang, phase1_res)
        ast1_dict = phase2_res.get('ast1_dict')
        ast2_dict = phase2_res.get('ast2_dict')
        print(f"   ✅ شباهت ساختاری: {phase2_res.get('ast_similarity_score', 0):.2f}%")
        print(f"   📊 گره‌های AST: {phase2_res.get('ast_statistics', {}).get('code1', {}).get('total_nodes', 0)} و {phase2_res.get('ast_statistics', {}).get('code2', {}).get('total_nodes', 0)}")

    # ---------- فاز ۳ ----------
    phase3_res = None
    if PHASE3_OK:
        print("\n[فاز ۳] تحلیل گراف جریان کنترل (CFG) ...")
        analyzer3 = Phase3CFGSimilarity()
        # ارسال AST واقعی از طریق phase2_res
        phase3_res = analyzer3.analyze_code_pair(
            code1, code2,
            phase1_results=phase1_res,
            phase2_results=phase2_res   # حاوی ast1_dict و ast2_dict است
        )
        print(f"   ✅ شباهت رفتاری: {phase3_res.get('cfg_similarity_score', 0):.2f}%")
        if 'combined_similarity_score' in phase3_res:
            print(f"   🎯 نمره ترکیبی: {phase3_res['combined_similarity_score']:.2f}%")

    # ---------- ترکیب نتایج ----------
    final = {
        'code1': args.code1,
        'code2': args.code2,
        'language': args.lang,
        'phases_executed': {
            'phase1': phase1_res is not None,
            'phase2': phase2_res is not None,
            'phase3': phase3_res is not None
        }
    }

    # استخراج نمرات
    token_score = phase1_res.get('overall_similarity', 0) / 100 if phase1_res else 0.0
    ast_score = phase2_res.get('ast_similarity_score', 0) / 100 if phase2_res else 0.0
    cfg_score = phase3_res.get('cfg_similarity_score', 0) / 100 if phase3_res else 0.0

    weights = {'token': 0.2, 'ast': 0.3, 'cfg': 0.5}
    combined = (weights['token'] * token_score +
                weights['ast'] * ast_score +
                weights['cfg'] * cfg_score) * 100

    final['scores'] = {
        'token': token_score * 100,
        'ast': ast_score * 100,
        'cfg': cfg_score * 100,
        'combined': combined,
        'weights': weights
    }

    # تشخیص سرقت
    threshold = 0.65
    is_plagiarism = combined >= (threshold * 100)
    final['verdict'] = {
        'threshold': threshold * 100,
        'is_plagiarism': is_plagiarism,
        'decision': 'PLAGIARISM_SUSPECTED' if is_plagiarism else 'CLEAN',
        'confidence': min(combined / 100, 1.0) * 100
    }

    # ذخیره گزارش نهایی
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("📊  گزارش نهایی")
    print("=" * 70)
    print(f"توکن:     {final['scores']['token']:.2f}%")
    print(f"ساختار:   {final['scores']['ast']:.2f}%")
    print(f"رفتار:    {final['scores']['cfg']:.2f}%")
    print(f"ترکیبی:   {final['scores']['combined']:.2f}%")
    print("-" * 70)
    print(f"تشخیص:    {final['verdict']['decision']}")
    print(f"اطمینان:  {final['verdict']['confidence']:.1f}%")
    print("=" * 70)
    print(f"\n📄 گزارش کامل در {args.output} ذخیره شد.")


if __name__ == '__main__':
    main()