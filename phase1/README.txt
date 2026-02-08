# to build the grammers based on the lexers: 

cd ./phase1/grammers 

java -jar antlr-4.13.2-complete.jar -Dlanguage=Python3 -o ./generated/python Python3Lexer.g4

java -jar antlr-4.13.2-complete.jar -Dlanguage=Python3 -o ./generated/java JavaLexer.g4

java -jar antlr-4.13.2-complete.jar -Dlanguage=Python3 -o ./generated/cpp CPP14Lexer.g4