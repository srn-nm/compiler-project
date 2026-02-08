"""
Factory برای ایجاد handlerهای زبان‌های مختلف
"""

from typing import Dict, Any, Optional
from .python_handler import PythonHandler
from .java_handler import JavaHandler
from .cpp_handler import CppHandler  
from .base_handler import BaseLanguageHandler

class LanguageHandlerFactory:
    """Factory برای ایجاد handlerهای زبان‌های مختلف"""
    
    @staticmethod
    def create_handler(language: str, config: Dict[str, Any]) -> BaseLanguageHandler:
        """
        ایجاد handler برای زبان مورد نظر
        
        Args:
            language: نام زبان ('python', 'java', 'cpp', 'c')
            config: تنظیمات پیکربندی
            
        Returns:
            نمونه‌ای از BaseLanguageHandler
        """
        language = language.lower()
        
        if language in ['python', 'py']:
            return PythonHandler(config)
        elif language in ['java', 'jav']:
            return JavaHandler(config)
        elif language in ['cpp', 'c++', 'cplusplus']:
            return CppHandler(config)
        elif language in ['c']:
            # C و C++ شبیه هستند، می‌توانیم از CppHandler استفاده کنیم
            # یا اگر تفاوت‌های زیادی دارند، CHandler جدا بسازیم
            return CppHandler(config)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    @staticmethod
    def detect_language(code: str) -> str:
        """
        تشخیص خودکار زبان کد
        
        Args:
            code: کد منبع
            
        Returns:
            نام زبان تشخیص داده شده
        """
        lines = code.strip().split('\n')
        
        # بررسی الگوهای زبان‌های مختلف
        for line in lines[:10]:  # فقط ۱۰ خط اول را بررسی کن
            line = line.strip()
            
            # پایتون
            if (line.startswith('def ') or line.startswith('class ') or 
                line.startswith('import ') or line.startswith('from ') or
                line.startswith('print(')):
                return 'python'
            
            # جاوا
            elif (line.startswith('public ') or line.startswith('private ') or 
                  line.startswith('protected ') or 'class ' in line and '{' in line or
                  line.startswith('import java.')):
                return 'java'
            
            # ++C
            elif (line.startswith('#include <') or 
                  line.startswith('using namespace') or
                  'std::' in line or
                  line.startswith('template<')):
                return 'cpp'
            
            # C
            elif (line.startswith('#include "') and '.h"' in line or
                  line.startswith('#define ') or
                  line.startswith('typedef ')):
                return 'c'
        
        # پیش‌فرض
        return 'python'
    
    @staticmethod
    def get_supported_languages() -> List[str]:
        return ['python', 'java', 'cpp', 'c']
    
    @staticmethod
    def get_language_extensions() -> Dict[str, List[str]]:
        """دریافت پسوندهای فایل برای هر زبان"""
        return {
            'python': ['.py', '.pyw', '.pyc'],
            'java': ['.java', '.jar'],
            'cpp': ['.cpp', '.cxx', '.cc', '.c++', '.hpp', '.hxx', '.hh'],
            'c': ['.c', '.h']
        }