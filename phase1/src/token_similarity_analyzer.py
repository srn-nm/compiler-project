from antlr4 import *
from grammers.generated.python.Python3Lexer import Python3Lexer
from collections import Counter
import math

class TokenSimilarityAnalyzer:
    def __init__(self):
        self.lexer = None
        
    def tokenize_code(self, code):
        """Convert Python code to token stream"""
        input_stream = InputStream(code)
        self.lexer = Python3Lexer(input_stream)
        token_stream = CommonTokenStream(self.lexer)
        token_stream.fill()
        
        # Filter out whitespace and comments
        tokens = []
        for token in token_stream.tokens:
            if token.type != -1:  # Skip EOF
                token_name = self.lexer.symbolicNames[token.type]
                # Skip hidden tokens (whitespace, newlines)
                if token.channel != Token.HIDDEN_CHANNEL:
                    tokens.append({
                        'type': token_name,
                        'text': token.text,
                        'line': token.line,
                        'column': token.column
                    })
        return tokens
    
    def calculate_similarity(self, code1, code2):
        """Calculate token-based similarity between two code snippets"""
        # Tokenize both codes
        tokens1 = self.tokenize_code(code1)
        tokens2 = self.tokenize_code(code2)
        
        # Extract token types for comparison
        token_types1 = [token['type'] for token in tokens1]
        token_types2 = [token['type'] for token in tokens2]
        
        # Calculate various similarity metrics
        metrics = {
            'jaccard_similarity': self.jaccard_similarity(token_types1, token_types2),
            'cosine_similarity': self.cosine_similarity(token_types1, token_types2),
            'levenshtein_similarity': self.levenshtein_similarity(token_types1, token_types2),
            'token_overlap': self.token_overlap(tokens1, tokens2),
            'sequence_similarity': self.sequence_similarity(token_types1, token_types2)
        }
        
        # Overall similarity score (weighted average)
        weights = {
            'jaccard_similarity': 0.2,
            'cosine_similarity': 0.3,
            'levenshtein_similarity': 0.3,
            'sequence_similarity': 0.2
        }
        
        overall_score = sum(metrics[key] * weights.get(key, 0) 
                          for key in weights if key in metrics)
        
        return {
            'overall_similarity': overall_score * 100,  # Convert to percentage
            'metrics': metrics,
            'token_counts': {
                'code1': len(tokens1),
                'code2': len(tokens2),
                'common': len(set(token_types1) & set(token_types2))
            },
            'matched_sections': self.find_matching_sections(tokens1, tokens2)
        }
    
    def jaccard_similarity(self, list1, list2):
        """Jaccard similarity between two token sets"""
        set1 = set(list1)
        set2 = set(list2)
        if not set1 and not set2:
            return 1.0
        return len(set1 & set2) / len(set1 | set2)
    
    def cosine_similarity(self, list1, list2):
        """Cosine similarity between token frequency vectors"""
        vec1 = Counter(list1)
        vec2 = Counter(list2)
        
        # Get all unique tokens
        all_tokens = set(vec1.keys()) | set(vec2.keys())
        
        # Create vectors
        v1 = [vec1.get(token, 0) for token in all_tokens]
        v2 = [vec2.get(token, 0) for token in all_tokens]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def levenshtein_similarity(self, list1, list2):
        """Levenshtein distance similarity (edit distance)"""
        # Create matrix
        len1, len2 = len(list1), len(list2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Initialize matrix
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        # Fill matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if list1[i-1] == list2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        
        # Calculate similarity
        max_len = max(len1, len2)
        if max_len == 0:
            return 1.0
        return 1 - (dp[len1][len2] / max_len)
    
    def sequence_similarity(self, list1, list2):
        """Sequence similarity using longest common subsequence"""
        # Find LCS length
        len1, len2 = len(list1), len(list2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if list1[i-1] == list2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[len1][len2]
        max_len = max(len1, len2)
        
        if max_len == 0:
            return 1.0
        return lcs_length / max_len
    
    def token_overlap(self, tokens1, tokens2):
        """Find overlapping token sequences"""
        # Convert to text sequences for matching
        seq1 = [token['text'] for token in tokens1]
        seq2 = [token['text'] for token in tokens2]
        
        overlaps = []
        min_len = min(len(seq1), len(seq2))
        
        # Find matching subsequences
        for length in range(min_len, 2, -1):  # Look for sequences of at least 3 tokens
            for i in range(len(seq1) - length + 1):
                subseq = seq1[i:i+length]
                subseq_str = ' '.join(subseq)
                
                seq2_str = ' '.join(seq2)
                if subseq_str in seq2_str:
                    overlaps.append({
                        'tokens': subseq,
                        'start_pos1': i,
                        'end_pos1': i + length - 1,
                        'similarity': length / min_len
                    })
        
        return overlaps
    
    def find_matching_sections(self, tokens1, tokens2):
        """Identify exactly matching sections between two token sequences"""
        matches = []
        window_size = 5  # Look for matches of at least 5 consecutive tokens
        
        # Create text representations
        text1 = [f"{t['type']}:{t['text']}" for t in tokens1]
        text2 = [f"{t['type']}:{t['text']}" for t in tokens2]
        
        # Find common subsequences
        for i in range(len(text1) - window_size + 1):
            for j in range(len(text2) - window_size + 1):
                match_length = 0
                while (i + match_length < len(text1) and 
                       j + match_length < len(text2) and 
                       text1[i + match_length] == text2[j + match_length]):
                    match_length += 1
                
                if match_length >= window_size:
                    matches.append({
                        'start1': i,
                        'end1': i + match_length - 1,
                        'start2': j,
                        'end2': j + match_length - 1,
                        'length': match_length,
                        'tokens': text1[i:i+match_length]
                    })
        
        return matches