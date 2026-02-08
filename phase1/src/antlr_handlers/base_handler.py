from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from antlr4 import InputStream, CommonTokenStream

"""base handler برای مدیریت مشترک همه زبان‌ها"""

class BaseLanguageHandler(ABC):
    """کلاس پایه برای هندلرهای زبان‌های مختلف"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lexer_class = None
        self.parser_class = None
    
    @abstractmethod
    def get_tokens(self, code: str) -> List[str]:
        """استخراج توکن‌ها از کد"""
        pass
    
    @abstractmethod
    def get_normalized_tokens(self, code: str) -> List[str]:
        """استخراج توکن‌های نرمال‌سازی شده"""
        pass
    
    def preprocess_code(self, code: str) -> str:
        """پیش‌پردازش کد قبل از توکن‌سازی"""
        if self.config.get('remove_comments', True):
            code = self._remove_comments(code)
        
        if self.config.get('normalize_whitespace', True):
            code = self._normalize_whitespace(code)
        
        if not self.config.get('case_sensitive', False):
            code = code.lower()
        
        return code
    
    def _remove_comments(self, code: str) -> str:
        """حذف کامنت‌ها (پیاده‌سازی وابسته به زبان در کلاس فرزند)"""
        return code
    
    def _normalize_whitespace(self, code: str) -> str:
        """نرمال‌سازی فاصله‌ها"""
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.rstrip() 
            if line.strip():  
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _create_token_stream(self, code: str):
        """ایجاد token stream از کد"""
        if not self.lexer_class:
            raise NotImplementedError("Lexer class not defined")
        
        input_stream = InputStream(code)
        lexer = self.lexer_class(input_stream)
        
        if hasattr(lexer, 'setOptions'):
            options = self.config.get('lexer_options', {})
            for key, value in options.items():
                if hasattr(lexer, key):
                    setattr(lexer, key, value)
        
        return CommonTokenStream(lexer)