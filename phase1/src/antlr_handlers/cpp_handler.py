import re
from typing import List, Dict, Any, Optional
from antlr4 import InputStream, CommonTokenStream

from .base_handler import BaseLanguageHandler

# اضافه کردن مسیر گرامرهای تولید شده
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../grammars/generated/cpp'))

try:
    from CPP14Lexer import CPP14Lexer
    from CPP14Parser import CPP14Parser
    ANTLR_AVAILABLE = True
except ImportError:
    ANTLR_AVAILABLE = False
    print("⚠️  Warning: ANTLR generated files for C++ not found. Using fallback tokenizer.")

class CppHandler(BaseLanguageHandler):
    """Handler برای زبان ++C"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if ANTLR_AVAILABLE:
            self.lexer_class = CPP14Lexer
            self.parser_class = CPP14Parser
        else:
            print("Using regex-based fallback for C++")
        
        # کلمات کلیدی مخصوص ++C
        self.cpp_keywords = {
            'auto', 'break', 'case', 'char', 'const', 'continue', 'default',
            'do', 'double', 'else', 'enum', 'extern', 'float', 'for', 'goto',
            'if', 'int', 'long', 'register', 'return', 'short', 'signed',
            'sizeof', 'static', 'struct', 'switch', 'typedef', 'union',
            'unsigned', 'void', 'volatile', 'while', 'cout', 'cin', 'endl',
            'string', 'vector', 'map', 'set', 'include', 'using', 'namespace',
            'std', 'main', 'class', 'public', 'private', 'protected', 'template',
            'typename', 'new', 'delete', 'true', 'false', 'nullptr', 'bool'
        }
        
        # عملگرهای مخصوص ++C
        self.cpp_operators = {
            '<<', '>>', '++', '--', '==', '!=', '<=', '>=', '&&', '||',
            '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=',
            '->', '::', '.*', '->*'
        }
    
    def get_tokens(self, code: str) -> List[str]:
        """استخراج توکن‌ها با ANTLR یا fallback"""
        if not ANTLR_AVAILABLE:
            return self._get_tokens_fallback(code)
        
        # پیش‌پردازش کد
        processed_code = self.preprocess_code(code)
        
        # ایجاد token stream با ANTLR
        input_stream = InputStream(processed_code)
        lexer = self.lexer_class(input_stream)
        stream = CommonTokenStream(lexer)
        stream.fill()
        
        # استخراج توکن‌ها
        tokens = []
        for token in stream.tokens:
            if token.type != token.EOF:
                token_text = token.text.strip()
                if token_text:
                    tokens.append(token_text)
        
        return tokens
    
    def get_normalized_tokens(self, code: str) -> List[str]:
        """استخراج توکن‌های نرمال‌سازی شده"""
        tokens = self.get_tokens(code)
        
        if self.config.get('normalize_identifiers', True):
            tokens = self._normalize_identifiers(tokens)
        
        if self.config.get('normalize_literals', True):
            tokens = self._normalize_literals(tokens)
        
        if self.config.get('normalize_headers', True):
            tokens = self._normalize_headers(tokens)
        
        return tokens
    
    def _get_tokens_fallback(self, code: str) -> List[str]:
        """Fallback tokenizer برای ++C با regex"""
        processed_code = self.preprocess_code(code)
        
        # الگوهای regex مخصوص ++C
        patterns = [
            # Directives و Headers
            r'#\s*(include|define|ifdef|ifndef|endif|pragma)\b',
            r'<[^>]+>',  # system headers: <iostream>
            r'"[^"]+"',  # user headers: "myheader.h"
            
            # Keywords
            r'\b(auto|break|case|char|const|continue|default|do|double|else|enum|extern|float|for|goto|if|int|long|register|return|short|signed|sizeof|static|struct|switch|typedef|union|unsigned|void|volatile|while|class|public|private|protected|template|typename|namespace|using|new|delete|true|false|nullptr|bool)\b',
            
            # Types و Standard Library
            r'\b(cout|cin|endl|cerr|clog|string|vector|map|set|list|deque|array|tuple|pair|shared_ptr|unique_ptr|function)\b',
            r'\b(std|endl)\b',
            
            # Identifiers
            r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',
            
            # Operators
            r'::|->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|\.\*|->\*',
            r'[+\-*/%&|^~<>!=]=?',
            
            # Literals
            r'\b\d+(\.\d+)?([eE][+-]?\d+)?(f|F|l|L)?\b',  # numbers
            r'\b0x[0-9a-fA-F]+\b',  # hex
            r'\b0[0-7]+\b',  # octal
            r'L?"[^"]*"',  # strings
            r"L?'[^']'",  # chars
            
            # Delimiters
            r'[\[\]{}();,:\.]',
        ]
        
        combined_pattern = '|'.join(f'({p})' for p in patterns)
        tokens = []
        
        for match in re.finditer(combined_pattern, processed_code):
            token = match.group()
            if token and not token.isspace():
                tokens.append(token)
        
        return tokens
    
    def _remove_comments(self, code: str) -> str:
        """حذف کامنت‌های ++C"""
        # حذف کامنت‌های چندخطی /* */
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        
        # حذف کامنت‌های تک خطی //
        lines = code.split('\n')
        cleaned_lines = []
        
        in_multiline_comment = False
        multiline_buffer = []
        
        for line in lines:
            # بررسی برای کامنت‌های چندخطی در یک خط
            if '/*' in line and '*/' in line:
                # کامنت در وسط خط
                line = re.sub(r'/\*.*?\*/', '', line)
            elif '/*' in line:
                # شروع کامنت چندخطی
                in_multiline_comment = True
                line = line.split('/*')[0]
                multiline_buffer.append(line)
                continue
            elif '*/' in line:
                # پایان کامنت چندخطی
                in_multiline_comment = False
                line = line.split('*/')[1]
                # اضافه کردن خطوط ذخیره شده
                if multiline_buffer:
                    cleaned_lines.extend(multiline_buffer)
                    multiline_buffer = []
            
            if not in_multiline_comment:
                # حذف کامنت‌های تک خطی
                if '//' in line:
                    line = line.split('//')[0]
                
                if line.strip():
                    cleaned_lines.append(line)
            else:
                # در حالتی که در کامنت چندخطی هستیم
                multiline_buffer.append('')  # خط خالی
            
        return '\n'.join(cleaned_lines)
    
    def _normalize_identifiers(self, tokens: List[str]) -> List[str]:
        """نرمال‌سازی شناسه‌های ++C"""
        normalized = []
        
        for token in tokens:
            # بررسی آیا شناسه است (نه keyword، نه operator، نه literal)
            if (token not in self.cpp_keywords and 
                token not in self.cpp_operators and
                not re.match(r'^[#<>"\'\.:;,\[\]{}()]$', token) and
                not re.match(r'^\d', token) and  # نه عدد
                not (token.startswith('"') and token.endswith('"')) and  # نه رشته
                not (token.startswith("'") and token.endswith("'")) and  # نه کاراکتر
                re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token)):  # فرمت identifier
            
                # بررسی برای نام‌های استاندارد که نباید نرمال شوند
                std_names = {'cout', 'cin', 'endl', 'string', 'vector', 'main'}
                if token not in std_names:
                    normalized.append('IDENTIFIER')
                else:
                    normalized.append(token)
            else:
                normalized.append(token)
        
        return normalized
    
    def _normalize_literals(self, tokens: List[str]) -> List[str]:
        """نرمال‌سازی مقادیر ثابت ++C"""
        normalized = []
        
        for token in tokens:
            # اعداد
            if re.match(r'^\d+(\.\d+)?([eE][+-]?\d+)?(f|F|l|L)?$', token):
                normalized.append('NUMBER')
            elif re.match(r'^0x[0-9a-fA-F]+$', token):  # hexadecimal
                normalized.append('NUMBER')
            elif re.match(r'^0[0-7]+$', token):  # octal
                normalized.append('NUMBER')
            
            # رشته‌ها
            elif re.match(r'^L?"[^"]*"$', token):
                normalized.append('STRING')
            
            # کاراکترها
            elif re.match(r"^L?'[^']'$", token):
                normalized.append('CHAR')
            
            # boolean و null
            elif token in ['true', 'false']:
                normalized.append('BOOLEAN')
            elif token in ['nullptr', 'NULL']:
                normalized.append('NULL')
            
            else:
                normalized.append(token)
        
        return normalized
    
    def _normalize_headers(self, tokens: List[str]) -> List[str]:
        """نرمال‌سازی هدرها و directives"""
        normalized = []
        
        for token in tokens:
            # #include directives
            if token.startswith('#include'):
                normalized.append('#include')
                continue
            
            # system headers: <iostream> -> <HEADER>
            if token.startswith('<') and token.endswith('>'):
                normalized.append('<HEADER>')
                continue
            
            # user headers: "myheader.h" -> "HEADER"
            if token.startswith('"') and token.endswith('"') and '.h' in token:
                normalized.append('"HEADER"')
                continue
            
            # سایر directives
            if token.startswith('#'):
                directive = token.split()[0] if ' ' in token else token
                normalized.append(directive)
                continue
            
            normalized.append(token)
        
        return normalized
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """استخراج توابع از کد ++C (برای فازهای بعدی)"""
        functions = []
        
        # الگوی ساده برای تشخیص توابع
        pattern = r'\b(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{'
        
        for match in re.finditer(pattern, code, re.MULTILINE):
            return_type = match.group(1)
            function_name = match.group(2)
            params = match.group(3).strip()
            
            functions.append({
                'name': function_name,
                'return_type': return_type,
                'parameters': params,
                'line': code[:match.start()].count('\n') + 1
            })
        
        return functions
    
    def get_token_statistics(self, code: str) -> Dict[str, Any]:
        """آمار توکن‌های ++C"""
        tokens = self.get_tokens(code)
        normalized_tokens = self.get_normalized_tokens(code)
        
        # شمارش انواع توکن
        counts = {
            'total': len(tokens),
            'unique': len(set(tokens)),
            'keywords': sum(1 for t in tokens if t in self.cpp_keywords),
            'identifiers': sum(1 for t in normalized_tokens if t == 'IDENTIFIER'),
            'numbers': sum(1 for t in normalized_tokens if t == 'NUMBER'),
            'strings': sum(1 for t in normalized_tokens if t in ['STRING', 'CHAR']),
            'operators': sum(1 for t in tokens if t in self.cpp_operators or 
                           re.match(r'^[+\-*/%&|^~<>!=]=?$', t)),
            'headers': sum(1 for t in tokens if t.startswith('#include')),
            'directives': sum(1 for t in tokens if t.startswith('#') and not t.startswith('#include'))
        }
        
        # محبوب‌ترین توکن‌ها
        from collections import Counter
        token_counter = Counter(tokens)
        most_common = token_counter.most_common(10)
        
        return {
            'counts': counts,
            'most_common': most_common,
            'sample_tokens': tokens[:20]
        }