pip install -r requirements.txt

source venv/bin/activate 


# to build the grammers based on the lexers: 

cd ./phase1/grammers 

java -jar antlr-4.13.2-complete.jar -Dlanguage=Python3 -o ./generated/python Python3Lexer.g4
java -jar antlr-4.13.2-complete.jar -Dlanguage=Python3 -o ./generated/python Python3Parser.g4

java -jar antlr-4.13.2-complete.jar -Dlanguage=Python3 -o ./generated/java JavaLexer.g4

java -jar antlr-4.13.2-complete.jar -Dlanguage=Python3 -o ./generated/cpp CPP14Lexer.g4




# تحلیل دو فایل پایتون
python phase1/src/main.py phase1/tests/test_python/code1.py phase1/tests/test_python/code2.py --verbose
python -m phase1.src.main phase1/tests/test_python/code1.py phase1/tests/test_python/code2.py --verbose

# با خروجی HTML و نمایش جزئیات
python phase1/src/main.py phase1/tests/test_python/code1.py phase1/tests/test_python/code2.py --visual --verbose

# با فرمت خاص
python phase1/src/main.py phase1/tests/test_python/code1.py phase1/tests/test_python/code2.py --format html

# ماتریس مقایسه چند فایل
python phase1/src/main.py *.py --matrix --output similarity_matrix