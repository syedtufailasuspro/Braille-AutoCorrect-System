from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import time
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from enum import Enum
import heapq
import re
import os

# Import your existing Braille system classes
@dataclass
class Suggestion:
    """Represents a correction suggestion with metadata"""
    word: str
    confidence: float
    distance: int
    frequency: int
    source: str  # "exact", "fuzzy", "learned"
    number: int  # Suggestion number (1-5)

class BrailleInputProcessor:
    """Handles Braille QWERTY input conversion to English letters"""
    
    # Custom mapping of QWERTY keys to Braille characters
    BRAILLE_TO_LETTER = {
        'd': 'a',             # dot 1
        'dw': 'b',            # dots 1,2
        'dk': 'c',            # dots 1,4
        'dko': 'd',           # dots 1,4,5
        'do': 'e',            # dots 1,5
        'dkw': 'f',           # dots 1,2,4
        'dkow': 'g',          # dots 1,2,4,5
        'dow': 'h',           # dots 1,2,5
        'kw': 'i',            # dots 2,4
        'kow': 'j',           # dots 2,4,5
        'dq': 'k',            # dots 1,3
        'dqw': 'l',           # dots 1,2,3
        'dkq': 'm',           # dots 1,3,4
        'dkoq': 'n',          # dots 1,3,4,5
        'doq': 'o',           # dots 1,3,5
        'dkqw': 'p',          # dots 1,2,3,4
        'dkoqw': 'q',         # dots 1,2,3,4,5
        'doqw': 'r',          # dots 1,2,3,5
        'kqw': 's',           # dots 2,3,4
        'koqw': 't',          # dots 2,3,4,5
        'dpq': 'u',           # dots 1,3,6
        'dpqw': 'v',          # dots 1,2,3,6
        'kopw': 'w',          # dots 2,4,5,6
        'dkpq': 'x',          # dots 1,3,4,6
        'dkopq': 'y',         # dots 1,3,4,5,6
        'dopq': 'z'           # dots 1,3,5,6
    }
    
    @staticmethod
    def convert_braille_to_text(braille_input: str) -> str:
        """Convert Braille QWERTY input to English text"""
        if not braille_input.strip():
            return ""
        
        # Sort the characters for consistent lookup (e.g., 'dw' == 'wd')
        sorted_input = ''.join(sorted(braille_input.lower().strip()))
        letter = BrailleInputProcessor.BRAILLE_TO_LETTER.get(sorted_input, '?')
        return letter
    
    @staticmethod
    def convert_braille_sequence_to_word(braille_sequence: str) -> str:
        """Convert a sequence of Braille inputs separated by spaces to a word"""
        if not braille_sequence.strip():
            return ""
        
        braille_letters = braille_sequence.strip().split()
        english_letters = []
        
        for braille_letter in braille_letters:
            if braille_letter:  # Skip empty strings
                letter = BrailleInputProcessor.convert_braille_to_text(braille_letter)
                english_letters.append(letter)
        
        return ''.join(english_letters)

class TrieNode:
    """Trie node for efficient prefix matching"""
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = None
        self.frequency = 0

class BrailleTrie:
    """Optimized Trie for Braille patterns with fuzzy matching"""
    
    def __init__(self):
        self.root = TrieNode()
        self.word_frequencies = defaultdict(int)
    
    def insert(self, word: str, frequency: int = 1):
        """Insert word into trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.word = word
        node.frequency = frequency
        self.word_frequencies[word] = frequency
    
    def search_exact(self, pattern: str) -> Optional[str]:
        """Exact pattern search"""
        node = self.root
        for char in pattern:
            if char not in node.children:
                return None
            node = node.children[char]
        return node.word if node.is_word else None

class BKTree:
    """BK-Tree for efficient fuzzy string matching"""
    
    def __init__(self, distance_func):
        self.tree = {}
        self.distance_func = distance_func
    
    def add(self, word: str):
        """Add word to BK-Tree"""
        if not self.tree:
            self.tree = {"word": word, "children": {}}
            return
        
        self._add_recursive(self.tree, word)
    
    def _add_recursive(self, node, word):
        """Recursively add word to tree"""
        current_word = node["word"]
        distance = self.distance_func(word, current_word)
        
        # Convert float distance to integer for dictionary key
        int_distance = int(round(distance))
        
        if int_distance in node["children"]:
            self._add_recursive(node["children"][int_distance], word)
        else:
            node["children"][int_distance] = {"word": word, "children": {}}
    
    def search(self, word: str, max_distance: int) -> List[Tuple[str, int]]:
        """Search for words within max_distance"""
        if not self.tree:
            return []
        
        results = []
        self._search_recursive(self.tree, word, max_distance, results)
        return results
    
    def _search_recursive(self, node, target, max_distance, results):
        """Recursively search tree"""
        current_word = node["word"]
        distance = self.distance_func(target, current_word)
        
        if distance <= max_distance:
            results.append((current_word, int(distance)))
        
        # Convert float distances to integers for range function
        int_distance = int(round(distance))
        for child_distance in range(max(0, int_distance - max_distance), 
                                   int_distance + max_distance + 1):
            if child_distance in node["children"]:
                self._search_recursive(node["children"][child_distance], 
                                     target, max_distance, results)

class BrailleDistance:
    """Custom distance function optimized for Braille patterns"""
    
    @staticmethod
    def calculate(s1: str, s2: str) -> float:
        """Calculate weighted edit distance for Braille patterns"""
        if s1 == s2:
            return 0.0
        
        # Fallback to simple Levenshtein distance for stability
        return float(BrailleDistance._levenshtein(s1, s2))
    
    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Standard Levenshtein distance"""
        if len(s1) < len(s2):
            return BrailleDistance._levenshtein(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

class LearningEngine:
    """Machine learning component for adaptive suggestions"""
    
    def __init__(self):
        self.user_patterns = defaultdict(Counter)  # input -> selected word counts
        self.correction_history = []
        self.pattern_weights = defaultdict(float)
    
    def record_correction(self, input_pattern: str, selected_word: str):
        """Record user correction for learning"""
        self.user_patterns[input_pattern][selected_word] += 1
        self.correction_history.append((input_pattern, selected_word, time.time()))
        
        # Update pattern weights with temporal decay
        self._update_pattern_weights()
    
    def get_learned_suggestions(self, input_pattern: str) -> List[Tuple[str, float]]:
        """Get suggestions based on learned patterns"""
        if input_pattern not in self.user_patterns:
            return []
        
        suggestions = []
        total_count = sum(self.user_patterns[input_pattern].values())
        
        for word, count in self.user_patterns[input_pattern].items():
            confidence = count / total_count
            suggestions.append((word, confidence))
        
        return sorted(suggestions, key=lambda x: x[1], reverse=True)
    
    def _update_pattern_weights(self):
        """Update pattern weights with temporal decay"""
        current_time = time.time()
        decay_factor = 0.95
        time_threshold = 86400  # 24 hours
        
        for input_pattern, selected_word, timestamp in self.correction_history:
            age = current_time - timestamp
            if age < time_threshold:
                weight = decay_factor ** (age / 3600)  # Hourly decay
                self.pattern_weights[f"{input_pattern}->{selected_word}"] = weight

class BrailleAutoCorrect:
    """Main auto-correct system with Braille input support"""
    
    def __init__(self):
        self.trie = BrailleTrie()
        self.bk_tree = BKTree(BrailleDistance.calculate)
        self.learning_engine = LearningEngine()
        self.dictionary_loaded = False
        
        # Performance metrics
        self.stats = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'cache_hits': 0
        }
        
        # Simple caching
        self.cache = {}
        self.cache_size_limit = 1000
    
    def load_dictionary(self, words: List[str], frequencies: Dict[str, int] = None):
        """Load dictionary into data structures"""
        print(f"Loading {len(words)} words into dictionary...")
        
        if frequencies is None:
            frequencies = {word: 1 for word in words}
        
        start_time = time.time()
        
        # Load into Trie
        for word in words:
            freq = frequencies.get(word, 1)
            self.trie.insert(word, freq)
            self.bk_tree.add(word)
        
        self.dictionary_loaded = True
        load_time = time.time() - start_time
        print(f"Dictionary loaded in {load_time:.2f} seconds")
    
    def suggest_from_braille(self, braille_sequence: str, max_suggestions: int = 5, 
                           include_learned: bool = True) -> List[Suggestion]:
        """Get suggestions for Braille input sequence"""
        if not self.dictionary_loaded:
            raise ValueError("Dictionary not loaded. Call load_dictionary() first.")
        
        # Convert Braille sequence to English word
        english_word = BrailleInputProcessor.convert_braille_sequence_to_word(braille_sequence)
        
        # If conversion failed (contains '?'), return empty suggestions
        if '?' in english_word:
            return []
        
        # Get suggestions for the converted English word
        return self.suggest(english_word, max_suggestions, include_learned)
    
    def suggest(self, text_input: str, max_suggestions: int = 5, 
                include_learned: bool = True) -> List[Suggestion]:
        """Main suggestion method"""
        if not self.dictionary_loaded:
            raise ValueError("Dictionary not loaded. Call load_dictionary() first.")
        
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        # Check cache first
        cache_key = f"{text_input}:{max_suggestions}"
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            cached_suggestions = self.cache[cache_key]
            # Add numbers to cached suggestions
            for i, suggestion in enumerate(cached_suggestions[:max_suggestions]):
                suggestion.number = i + 1
            return cached_suggestions[:max_suggestions]
        
        pattern = text_input.lower().strip()
        suggestions = []
        
        # 1. Try exact match first
        exact_match = self.trie.search_exact(pattern)
        if exact_match:
            suggestions.append(Suggestion(
                word=exact_match,
                confidence=1.0,
                distance=0,
                frequency=self.trie.word_frequencies[exact_match],
                source="exact",
                number=1
            ))
        
        # 2. Get learned suggestions
        if include_learned:
            learned = self.learning_engine.get_learned_suggestions(pattern)
            for word, confidence in learned[:2]:  # Top 2 learned suggestions
                if word not in [s.word for s in suggestions]:
                    suggestions.append(Suggestion(
                        word=word,
                        confidence=confidence * 0.9,
                        distance=1,
                        frequency=self.trie.word_frequencies.get(word, 1),
                        source="learned",
                        number=len(suggestions) + 1
                    ))
        
        # 3. Fuzzy matching with BK-Tree
        max_distance = min(3, max(1, len(pattern) // 2))
        try:
            fuzzy_matches = self.bk_tree.search(pattern, max_distance)
            
            for word, distance in fuzzy_matches:
                if word not in [s.word for s in suggestions]:
                    confidence = max(0.1, 1.0 - (distance / (len(pattern) + 1)))
                    suggestions.append(Suggestion(
                        word=word,
                        confidence=confidence,
                        distance=int(distance),
                        frequency=self.trie.word_frequencies.get(word, 1),
                        source="fuzzy",
                        number=len(suggestions) + 1
                    ))
        except Exception as e:
            # Fallback to simple string matching if BK-Tree fails
            for word in list(self.trie.word_frequencies.keys())[:100]:
                distance = self._simple_edit_distance(pattern, word)
                if distance <= max_distance and word not in [s.word for s in suggestions]:
                    confidence = max(0.1, 1.0 - (distance / (len(pattern) + 1)))
                    suggestions.append(Suggestion(
                        word=word,
                        confidence=confidence,
                        distance=distance,
                        frequency=self.trie.word_frequencies.get(word, 1),
                        source="fuzzy",
                        number=len(suggestions) + 1
                    ))
        
        # 4. If no suggestions found, try a more lenient search
        if not suggestions:
            for word in list(self.trie.word_frequencies.keys())[:50]:
                distance = self._simple_edit_distance(pattern, word)
                if distance <= len(pattern):
                    confidence = max(0.05, 1.0 - (distance / (len(pattern) + len(word))))
                    suggestions.append(Suggestion(
                        word=word,
                        confidence=confidence,
                        distance=distance,
                        frequency=self.trie.word_frequencies.get(word, 1),
                        source="fuzzy",
                        number=len(suggestions) + 1
                    ))
        
        # 5. Rank suggestions
        suggestions = self._rank_suggestions(suggestions, pattern)
        
        # Limit results and assign numbers
        final_suggestions = suggestions[:max_suggestions]
        for i, suggestion in enumerate(final_suggestions):
            suggestion.number = i + 1
        
        # Cache results
        if len(self.cache) < self.cache_size_limit:
            self.cache[cache_key] = final_suggestions
        
        # Update stats
        response_time = time.time() - start_time
        self.stats['avg_response_time'] = (
            (self.stats['avg_response_time'] * (self.stats['total_queries'] - 1) + response_time)
            / self.stats['total_queries']
        )
        
        return final_suggestions
    
    def _simple_edit_distance(self, s1: str, s2: str) -> int:
        """Simple Levenshtein distance as fallback"""
        if len(s1) < len(s2):
            return self._simple_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _rank_suggestions(self, suggestions: List[Suggestion], pattern: str) -> List[Suggestion]:
        """Rank suggestions based on multiple factors"""
        def score(suggestion):
            # Base score from confidence
            base_score = suggestion.confidence
            
            # Frequency boost (normalize to 0-1 range)
            max_freq = max(s.frequency for s in suggestions) if suggestions else 1
            freq_score = suggestion.frequency / max_freq * 0.3
            
            # Distance penalty
            distance_penalty = suggestion.distance * 0.1
            
            # Source bonus
            source_bonus = {
                "exact": 0.5,
                "learned": 0.3,
                "fuzzy": 0.0
            }.get(suggestion.source, 0.0)
            
            return base_score + freq_score - distance_penalty + source_bonus
        
        return sorted(suggestions, key=score, reverse=True)
    
    def learn(self, input_pattern: str, selected_word: str):
        """Learn from user correction"""
        self.learning_engine.record_correction(input_pattern, selected_word)
        
        # Clear cache to ensure fresh results
        self.cache.clear()
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        cache_hit_rate = (self.stats['cache_hits'] / max(1, self.stats['total_queries'])) * 100
        return {
            **self.stats,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'dictionary_size': len(self.trie.word_frequencies)
        }

# Flask Application
app = Flask(__name__)
CORS(app)

# Global autocorrect instance
autocorrect = BrailleAutoCorrect()

# Sample dictionary for demo
SAMPLE_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just",
    "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
    "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us", "is", "water", "long", "find",
    "here", "thing", "every", "great", "where", "much", "before", "move", "right", "boy", "old", "too", "same", "tell", "say",
    "she", "may", "still", "such", "call", "came", "each", "part", "play", "small", "end", "why", "ask", "men", "read", "need",
    "land", "different", "home", "us", "move", "try", "kind", "hand", "picture", "again", "change", "off", "play", "spell",
    "air", "away", "animal", "house", "point", "page", "letter", "mother", "answer", "found", "study", "still", "learn",
    "should", "America", "world", "high", "every", "near", "add", "food", "between", "own", "below", "country", "plant",
    "school", "father", "keep", "tree", "never", "start", "city", "earth", "eye", "light", "thought", "head", "under",
    "story", "saw", "left", "don't", "few", "while", "along", "might", "close", "something", "seem", "next", "hard",
    "open", "example", "begin", "life", "always", "those", "both", "paper", "together", "got", "group", "often", "run"
]

def initialize_autocorrect(dictionary_file='word_frequencies.txt'):
    """Initialize the autocorrect system with dictionary from a file"""
    words = []
    frequencies = {}
    
    try:
        with open(dictionary_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0]
                    frequency = int(parts[1])
                    words.append(word)
                    frequencies[word] = frequency
                    
        autocorrect.load_dictionary(words, frequencies)
        print(f"Loaded dictionary with {len(words)} words")
    except FileNotFoundError:
        print(f"Dictionary file not found, using sample words")
        autocorrect.load_dictionary(SAMPLE_WORDS)
    except Exception as e:
        print(f"Error loading dictionary: {str(e)}, using sample words")
        autocorrect.load_dictionary(SAMPLE_WORDS)

# Initialize on startup
initialize_autocorrect('large_freq_dict.txt')

# Routes
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/convert-braille', methods=['POST'])
def convert_braille():
    """Convert Braille QWERTY input to English letter"""
    try:
        data = request.get_json()
        braille_input = data.get('input', '').strip()
        
        if not braille_input:
            return jsonify({'error': 'No input provided'}), 400
        
        english_letter = BrailleInputProcessor.convert_braille_to_text(braille_input)
        
        return jsonify({
            'braille_input': braille_input,
            'english_letter': english_letter,
            'valid': english_letter != '?'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/suggest-word', methods=['POST'])
def suggest_word():
    """Get suggestions for a Braille word sequence"""
    try:
        data = request.get_json()
        braille_sequence = data.get('braille_sequence', '').strip()
        max_suggestions = data.get('max_suggestions', 5)
        
        if not braille_sequence:
            return jsonify({'error': 'No braille sequence provided'}), 400
        
        # Convert Braille sequence to English word
        english_word = BrailleInputProcessor.convert_braille_sequence_to_word(braille_sequence)
        
        # Get suggestions
        suggestions = autocorrect.suggest_from_braille(braille_sequence, max_suggestions)
        
        # Convert suggestions to dictionaries for JSON serialization
        suggestions_data = [asdict(s) for s in suggestions]
        
        return jsonify({
            'braille_sequence': braille_sequence,
            'converted_word': english_word,
            'suggestions': suggestions_data,
            'count': len(suggestions_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/select-suggestion', methods=['POST'])
def select_suggestion():
    """Record user's suggestion selection for learning"""
    try:
        data = request.get_json()
        braille_sequence = data.get('braille_sequence', '').strip()
        selected_word = data.get('selected_word', '').strip()
        suggestion_number = data.get('suggestion_number', 0)
        
        if not braille_sequence or not selected_word:
            return jsonify({'error': 'Both braille_sequence and selected_word are required'}), 400
        
        # Convert Braille to English for learning
        english_word = BrailleInputProcessor.convert_braille_sequence_to_word(braille_sequence)
        
        # Learn from the selection
        autocorrect.learn(english_word, selected_word)
        
        return jsonify({
            'message': f'Selected suggestion #{suggestion_number}: "{selected_word}"',
            'braille_sequence': braille_sequence,
            'converted_word': english_word,
            'selected_word': selected_word,
            'suggestion_number': suggestion_number
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/build-sentence', methods=['POST'])
def build_sentence():
    """Add word to sentence being built"""
    try:
        data = request.get_json()
        current_sentence = data.get('current_sentence', [])
        new_word = data.get('new_word', '').strip()
        
        if not new_word:
            return jsonify({'error': 'No word provided'}), 400
        
        # Add the new word to the sentence
        updated_sentence = current_sentence + [new_word]
        
        return jsonify({
            'sentence': updated_sentence,
            'sentence_text': ' '.join(updated_sentence),
            'word_count': len(updated_sentence)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get system performance statistics"""
    try:
        stats = autocorrect.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)