## INSTALLATION

pip install -r requirements.txt

source venv/bin/activate 

# to build the grammers based on the lexers: 

cd ./phase1/grammers 

java -jar antlr-4.13.2-complete.jar -Dlanguage=Python3 -o ./generated/python Python3Lexer.g4
java -jar antlr-4.13.2-complete.jar -Dlanguage=Python3 -o ./generated/python Python3Parser.g4
java -jar antlr-4.13.2-complete.jar -Dlanguage=Python3 -o ./generated/java JavaLexer.g4
java -jar antlr-4.13.2-complete.jar -Dlanguage=Python3 -o ./generated/cpp CPP14Lexer.g4




## HOW TO RUN

# با خروجی HTML تحلیل دو فایل پایتون و نمایش جزئیات
python phase1/src/main.py phase1/tests/test_python/code1.py phase1/tests/test_python/code2.py --visual --verbose

# ماتریس مقایسه چند فایل
python phase1/src/main.py *.py --matrix --output similarity_matrix


After running the above command, results are saved in phase1/results