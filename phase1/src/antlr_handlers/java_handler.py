"""
Handler مخصوص زبان جاوا با استفاده از ANTLR
"""

import re
from typing import List, Dict, Any

from .base_handler import BaseLanguageHandler

# اضافه کردن مسیر گرامرهای تولید شده
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../grammars/generated/java'))

try:
    from JavaLexer import JavaLexer
    from JavaParser import JavaParser
    ANTLR_AVAILABLE = True
except ImportError:
    ANTLR_AVAILABLE = False

class JavaHandler(BaseLanguageHandler):
    """Handler برای زبان جاوا"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if ANTLR_AVAILABLE:
            self.lexer_class = JavaLexer
            self.parser_class = JavaParser
    
    def get_tokens(self, code: str) -> List[str]:
        """استخراج توکن‌ها"""
        if not ANTLR_AVAILABLE:
            return self._get_tokens_fallback(code)
        
        processed_code = self.preprocess_code(code)
        
        from antlr4 import InputStream, CommonTokenStream
        input_stream = InputStream(processed_code)
        lexer = self.lexer_class(input_stream)
        stream = CommonTokenStream(lexer)
        stream.fill()
        
        tokens = []
        for token in stream.tokens:
            if token.type != token.EOF:
                token_text = token.text.strip()
                if token_text:
                    tokens.append(token_text)
        
        return tokens
    
    def get_normalized_tokens(self, code: str) -> List[str]:
        """توکن‌های نرمال‌سازی شده"""
        tokens = self.get_tokens(code)
        
        if self.config.get('normalize_identifiers', True):
            tokens = self._normalize_identifiers(tokens)
        
        if self.config.get('normalize_literals', True):
            tokens = self._normalize_literals(tokens)
        
        return tokens
    
    def _get_tokens_fallback(self, code: str) -> List[str]:
        """Fallback برای جاوا"""
        processed_code = self.preprocess_code(code)
        
        patterns = [
            r'\b(public|private|protected|class|interface|extends|implements|static|void|int|String|boolean|if|else|for|while|return|new)\b',
            r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',
            r'[+\-*/%&|^~<>!=]=?',
            r'[\[\]{}();,.]',
            r'\b\d+\b',
            r'"[^"]*"',
        ]
        
        combined_pattern = '|'.join(f'({p})' for p in patterns)
        tokens = []
        
        for match in re.finditer(combined_pattern, processed_code):
            token = match.group()
            if token and not token.isspace():
                tokens.append(token)
        
        return tokens
    
    def _remove_comments(self, code: str) -> str:
        """حذف کامنت‌های جاوا"""
        # حذف کامنت‌های چندخطی
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        
        # حذف کامنت‌های تک خطی
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if '//' in line:
                line = line.split('//')[0]
            if line.strip():
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_identifiers(self, tokens: List[str]) -> List[str]:
        """نرمال‌سازی شناسه‌های جاوا"""
        normalized = []
        keywords = {
            'public', 'private', 'protected', 'class', 'interface',
            'extends', 'implements', 'static', 'void', 'int',
            'String', 'boolean', 'if', 'else', 'for', 'while',
            'return', 'new', 'this', 'super', 'try', 'catch',
            'finally', 'throw', 'throws'
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
            if re.match(r'^\d+$', token):
                normalized.append('NUMBER')
            elif re.match(r'^"[^"]*"$', token):
                normalized.append('STRING')
            elif token in ['true', 'false']:
                normalized.append('BOOLEAN')
            elif token == 'null':
                normalized.append('NULL')
            else:
                normalized.append(token)
        
        return normalized