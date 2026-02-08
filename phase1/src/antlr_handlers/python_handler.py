"""
Handler مخصوص زبان پایتون با استفاده از ANTLR
"""

import re
from typing import List, Dict, Any
from antlr4 import InputStream, CommonTokenStream

from .base_handler import BaseLanguageHandler

# ایمپورت lexer/parser تولید شده
import sys
import os

# اضافه کردن مسیر گرامرهای تولید شده به sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../grammars/generated/python'))

try:
    from Python3Lexer import Python3Lexer
    from Python3Parser import Python3Parser
    ANTLR_AVAILABLE = True
except ImportError:
    ANTLR_AVAILABLE = False
    print("Warning: ANTLR generated files not found. Using fallback tokenizer.")

class PythonHandler(BaseLanguageHandler):
    """Handler برای زبان پایتون"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if ANTLR_AVAILABLE:
            self.lexer_class = Python3Lexer
            self.parser_class = Python3Parser
        else:
            print("Using regex-based fallback for Python")
    
    def get_tokens(self, code: str) -> List[str]:
        """استخراج توکن‌ها با ANTLR"""
        if not ANTLR_AVAILABLE:
            return self._get_tokens_fallback(code)
        
        # پیش‌پردازش کد
        processed_code = self.preprocess_code(code)
        
        # ایجاد token stream
        input_stream = InputStream(processed_code)
        lexer = self.lexer_class(input_stream)
        stream = CommonTokenStream(lexer)
        stream.fill()
        
        # استخراج توکن‌ها
        tokens = []
        for token in stream.tokens:
            if token.type != token.EOF:
                token_text = token.text.strip()
                if token_text:  # اگر توکن خالی نبود
                    tokens.append(token_text)
        
        return tokens
    
    def get_normalized_tokens(self, code: str) -> List[str]:
        """استخراج توکن‌های نرمال‌سازی شده"""
        tokens = self.get_tokens(code)
        
        if self.config.get('normalize_identifiers', True):
            tokens = self._normalize_identifiers(tokens)
        
        if self.config.get('normalize_literals', True):
            tokens = self._normalize_literals(tokens)
        
        return tokens
    
    def _get_tokens_fallback(self, code: str) -> List[str]:
        """Fallback با regex اگر ANTLR در دسترس نباشد"""
        processed_code = self.preprocess_code(code)
        
        patterns = [
            r'\b(def|class|if|else|elif|for|while|return|import|from|as|try|except|finally|with|in|is|not|and|or)\b',  #keywords
            r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',  #identifiers
            r'[+\-*/%&|^~<>!=]=?',  #operators
            r'[\[\]{}():,.;@]',  #delimiters
            r'\b\d+(\.\d+)?([eE][+-]?\d+)?\b',  #numbers
            r'"(?:[^"\\]|\\.)*"',  #double quotes strings
            r"'(?:[^'\\]|\\.)*'",  #single quotes strings
        ]
        
        combined_pattern = '|'.join(f'({p})' for p in patterns)
        tokens = []
        
        for match in re.finditer(combined_pattern, processed_code):
            token = match.group()
            if token and not token.isspace():
                tokens.append(token)
        
        return tokens
    
    def _remove_comments(self, code: str) -> str:
        """حذف کامنت‌های پایتون"""
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # حذف کامنت‌های inline
            if '#' in line:
                line = line.split('#')[0]
            
            # حذف docstrings (ساده‌سازی شده)
            line = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', line)
            line = re.sub(r"\'\'\'[\s\S]*?\'\'\'", '', line)
            
            if line.strip():
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_identifiers(self, tokens: List[str]) -> List[str]:
        """نرمال‌سازی شناسه‌ها"""
        normalized = []
        keywords = {
            'def', 'class', 'if', 'else', 'elif', 'for', 'while',
            'return', 'import', 'from', 'as', 'try', 'except',
            'finally', 'with', 'in', 'is', 'not', 'and', 'or'
        }
        
        for token in tokens:
            if (token not in keywords and 
                re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token)):
                normalized.append('IDENTIFIER')
            else:
                normalized.append(token)
        
        return normalized
    
    def _normalize_literals(self, tokens: List[str]) -> List[str]:
        """نرمال‌سازی مقادیر ثابت"""
        normalized = []
        
        for token in tokens:
            if re.match(r'^\d+(\.\d+)?([eE][+-]?\d+)?$', token):
                normalized.append('NUMBER')
            elif re.match(r'^["\'].*["\']$', token):
                normalized.append('STRING')
            else:
                normalized.append(token)
        
        return normalized
    
    def get_token_types(self, code: str) -> List[Dict[str, Any]]:
        """دریافت نوع و مقدار توکن‌ها (برای دیباگ)"""
        if not ANTLR_AVAILABLE:
            return []
        
        processed_code = self.preprocess_code(code)
        input_stream = InputStream(processed_code)
        lexer = self.lexer_class(input_stream)
        stream = CommonTokenStream(lexer)
        stream.fill()
        
        token_info = []
        for token in stream.tokens:
            if token.type != token.EOF:
                token_info.append({
                    'text': token.text,
                    'type': token.type,
                    'type_name': lexer.symbolicNames[token.type] if token.type < len(lexer.symbolicNames) else str(token.type),
                    'line': token.line,
                    'column': token.column
                })
        
        return token_info