from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import time
import unicodedata
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from enum import Enum
import heapq
import re
import os

# Import your existing Braille system classes adapted for Hindi
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
    """Handles Hindi Bharati Braille QWERTY input conversion to Devanagari letters"""
    
    # Bharati Braille mapping for Hindi (simplified for common characters)
    HINDI_BRAILLE_TO_DEVANAGARI = {
        # Vowels
        'd': 'अ',              # a (dot 1)
        'do': 'आ',             # aa (dots 1,5)
        'dk': 'इ',             # i (dots 1,4)
        'dko': 'ई',            # ii (dots 1,4,5)
        'dp': 'उ',             # u (dots 1,6)
        'dpo': 'ऊ',            # uu (dots 1,5,6)
        'dpq': 'ए',            # e (dots 1,3,6)
        'doq': 'ओ',            # o (dots 1,3,5)
        
        # Consonants (basic set)
        'dw': 'क',             # ka (dots 1,2)
        'dkw': 'ख',            # kha (dots 1,2,4)
        'dkow': 'ग',           # ga (dots 1,2,4,5)
        'dow': 'घ',            # gha (dots 1,2,5)
        'kw': 'च',             # cha (dots 2,4)
        'kow': 'छ',            # chha (dots 2,4,5)
        'dq': 'ज',             # ja (dots 1,3)
        'dqw': 'झ',            # jha (dots 1,2,3)
        'dkq': 'ट',            # ta (dots 1,3,4)
        'dkoq': 'ठ',           # tha (dots 1,3,4,5)
        'kqw': 'ड',            # da (dots 2,3,4)
        'koqw': 'ढ',           # dha (dots 2,3,4,5)
        'dqp': 'त',            # ta (dots 1,3,6)
        'dqpw': 'थ',           # tha (dots 1,2,3,6)
        'kqp': 'द',            # da (dots 2,3,6)
        'kqpw': 'ध',           # dha (dots 2,3,6,4)
        'dkpq': 'न',           # na (dots 1,3,4,6)
        'dpw': 'प',            # pa (dots 1,2,6)
        'dpqw': 'फ',           # pha (dots 1,2,3,6)
        'kpw': 'ब',            # ba (dots 2,4,6)
        'kopw': 'भ',           # bha (dots 2,4,5,6)
        'dkp': 'म',            # ma (dots 1,4,6)
        'dkqw': 'य',           # ya (dots 1,2,3,4)
        'doqw': 'र',           # ra (dots 1,2,3,5)
        'dkoqw': 'ल',          # la (dots 1,2,3,4,5)
        'kopq': 'व',           # va (dots 2,3,4,5,6)
        'qw': 'श',             # sha (dots 2,3)
        'qpw': 'ष',            # shha (dots 2,3,6)
        'kq': 'स',             # sa (dots 2,3,4)
        'doww': 'ह',            # ha (dots 1,2,5)
        
        # Common conjuncts and special characters
        'dkowi': 'ं',           # anusvara (dots 1,2,4,5)
        'dpqi': 'ः',            # visarga (dots 1,3,6)
    }
    
    # Vowel signs (matras) - used after consonants
    VOWEL_SIGNS = {
        'o': 'ा',              # aa matra
        'k': 'ि',              # i matra  
        'ko': 'ी',             # ii matra
        'p': 'ु',              # u matra
        'po': 'ू',             # uu matra
        'pq': 'े',             # e matra
        'oq': 'ो',             # o matra
    }
    
    @staticmethod
    def normalize_hindi_text(text: str) -> str:
        """Normalize Hindi text for consistent processing"""
        return unicodedata.normalize('NFC', text.strip())
    
    @staticmethod
    def convert_braille_to_devanagari(braille_input: str) -> str:
        """Convert Braille QWERTY input to Devanagari character"""
        if not braille_input.strip():
            return ""
        
        # Sort the characters for consistent lookup
        sorted_input = ''.join(sorted(braille_input.lower().strip()))

        print(sorted_input)
        
        # First try full character mapping
        devanagari_char = BrailleInputProcessor.HINDI_BRAILLE_TO_DEVANAGARI.get(sorted_input)
        
        print(devanagari_char)
        if devanagari_char:
            return devanagari_char
        
        # Try vowel sign mapping
        vowel_sign = BrailleInputProcessor.VOWEL_SIGNS.get(sorted_input)
        if vowel_sign:
            return vowel_sign
            
        return '?'
    
    @staticmethod
    def convert_braille_sequence_to_word(braille_sequence: str) -> str:
        """Convert a sequence of Braille inputs to a Hindi word"""
        if not braille_sequence.strip():
            return ""
        
        braille_letters = braille_sequence.strip().split()
        devanagari_chars = []
        
        for braille_letter in braille_letters:
            if braille_letter:  # Skip empty strings
                char = BrailleInputProcessor.convert_braille_to_devanagari(braille_letter)
                devanagari_chars.append(char)
        
        word = ''.join(devanagari_chars)
        return BrailleInputProcessor.normalize_hindi_text(word)

class TrieNode:
    """Trie node for efficient prefix matching"""
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = None
        self.frequency = 0

class HindiBrailleTrie:
    """Optimized Trie for Hindi Braille patterns with fuzzy matching"""
    
    def __init__(self):
        self.root = TrieNode()
        self.word_frequencies = defaultdict(int)
    
    def insert(self, word: str, frequency: int = 1):
        """Insert Hindi word into trie"""
        normalized_word = BrailleInputProcessor.normalize_hindi_text(word)
        node = self.root
        for char in normalized_word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.word = normalized_word
        node.frequency = frequency
        self.word_frequencies[normalized_word] = frequency
    
    def search_exact(self, pattern: str) -> Optional[str]:
        """Exact pattern search"""
        normalized_pattern = BrailleInputProcessor.normalize_hindi_text(pattern)
        node = self.root
        for char in normalized_pattern:
            if char not in node.children:
                return None
            node = node.children[char]
        return node.word if node.is_word else None

class HindiBKTree:
    """BK-Tree for efficient fuzzy Hindi string matching"""
    
    def __init__(self, distance_func):
        self.tree = {}
        self.distance_func = distance_func
    
    def add(self, word: str):
        """Add Hindi word to BK-Tree"""
        normalized_word = BrailleInputProcessor.normalize_hindi_text(word)
        if not self.tree:
            self.tree = {"word": normalized_word, "children": {}}
            return
        
        self._add_recursive(self.tree, normalized_word)
    
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
        """Search for Hindi words within max_distance"""
        if not self.tree:
            return []
        
        normalized_word = BrailleInputProcessor.normalize_hindi_text(word)
        results = []
        self._search_recursive(self.tree, normalized_word, max_distance, results)
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

class HindiDistance:
    """Custom distance function optimized for Hindi Devanagari patterns"""
    
    @staticmethod
    def calculate(s1: str, s2: str) -> float:
        """Calculate weighted edit distance for Hindi patterns"""
        if s1 == s2:
            return 0.0
        
        # Normalize both strings
        s1 = BrailleInputProcessor.normalize_hindi_text(s1)
        s2 = BrailleInputProcessor.normalize_hindi_text(s2)
        
        # Standard Levenshtein distance with Hindi-specific considerations
        return float(HindiDistance._hindi_levenshtein(s1, s2))
    
    @staticmethod
    def _hindi_levenshtein(s1: str, s2: str) -> int:
        """Levenshtein distance with Hindi character considerations"""
        if len(s1) < len(s2):
            return HindiDistance._hindi_levenshtein(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                
                # Custom substitution cost for Hindi characters
                if c1 == c2:
                    substitution_cost = 0
                else:
                    # Lower cost for similar sounding consonants or vowel variations
                    substitution_cost = HindiDistance._get_substitution_cost(c1, c2)
                
                substitutions = previous_row[j] + substitution_cost
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def _get_substitution_cost(c1: str, c2: str) -> int:
        """Get custom substitution cost for Hindi characters"""
        # Define similar character groups
        similar_groups = [
            ['क', 'ख', 'ग', 'घ'],  # ka, kha, ga, gha
            ['च', 'छ', 'ज', 'झ'],  # cha, chha, ja, jha
            ['ट', 'ठ', 'ड', 'ढ'],  # ta, tha, da, dha
            ['त', 'थ', 'द', 'ध'],  # ta, tha, da, dha
            ['प', 'फ', 'ब', 'भ'],  # pa, pha, ba, bha
            ['अ', 'आ'],           # a, aa
            ['इ', 'ई'],           # i, ii
            ['उ', 'ऊ'],           # u, uu
        ]
        
        # Check if characters are in the same group
        for group in similar_groups:
            if c1 in group and c2 in group:
                return 1  # Lower cost for similar characters
        
        return 2  # Higher cost for dissimilar characters

class LearningEngine:
    """Machine learning component for adaptive Hindi suggestions"""
    
    def __init__(self):
        self.user_patterns = defaultdict(Counter)  # input -> selected word counts
        self.correction_history = []
        self.pattern_weights = defaultdict(float)
    
    def record_correction(self, input_pattern: str, selected_word: str):
        """Record user correction for learning"""
        normalized_input = BrailleInputProcessor.normalize_hindi_text(input_pattern)
        normalized_word = BrailleInputProcessor.normalize_hindi_text(selected_word)
        
        self.user_patterns[normalized_input][normalized_word] += 1
        self.correction_history.append((normalized_input, normalized_word, time.time()))
        
        # Update pattern weights with temporal decay
        self._update_pattern_weights()
    
    def get_learned_suggestions(self, input_pattern: str) -> List[Tuple[str, float]]:
        """Get suggestions based on learned patterns"""
        normalized_input = BrailleInputProcessor.normalize_hindi_text(input_pattern)
        
        if normalized_input not in self.user_patterns:
            return []
        
        suggestions = []
        total_count = sum(self.user_patterns[normalized_input].values())
        
        for word, count in self.user_patterns[normalized_input].items():
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

class HindiBrailleAutoCorrect:
    """Main Hindi auto-correct system with Braille input support"""
    
    def __init__(self):
        self.trie = HindiBrailleTrie()
        self.bk_tree = HindiBKTree(HindiDistance.calculate)
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
        """Load Hindi dictionary into data structures"""
        print(f"Loading {len(words)} Hindi words into dictionary...")
        
        if frequencies is None:
            frequencies = {word: 1 for word in words}
        
        start_time = time.time()
        
        # Load into Trie and BK-Tree
        for word in words:
            normalized_word = BrailleInputProcessor.normalize_hindi_text(word)
            freq = frequencies.get(word, 1)
            self.trie.insert(normalized_word, freq)
            self.bk_tree.add(normalized_word)
        
        self.dictionary_loaded = True
        load_time = time.time() - start_time
        print(f"Hindi dictionary loaded in {load_time:.2f} seconds")
    
    def suggest_from_braille(self, braille_sequence: str, max_suggestions: int = 5, 
                           include_learned: bool = True) -> List[Suggestion]:
        """Get suggestions for Hindi Braille input sequence"""
        if not self.dictionary_loaded:
            raise ValueError("Dictionary not loaded. Call load_dictionary() first.")
        
        # Convert Braille sequence to Hindi word
        hindi_word = BrailleInputProcessor.convert_braille_sequence_to_word(braille_sequence)
        
        # If conversion failed (contains '?'), return empty suggestions
        if '?' in hindi_word:
            return []
        
        # Get suggestions for the converted Hindi word
        return self.suggest(hindi_word, max_suggestions, include_learned)
    
    def suggest(self, text_input: str, max_suggestions: int = 5, 
                include_learned: bool = True) -> List[Suggestion]:
        """Main suggestion method for Hindi text"""
        if not self.dictionary_loaded:
            raise ValueError("Dictionary not loaded. Call load_dictionary() first.")
        
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        # Normalize input
        pattern = BrailleInputProcessor.normalize_hindi_text(text_input)
        
        # Check cache first
        cache_key = f"{pattern}:{max_suggestions}"
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            cached_suggestions = self.cache[cache_key]
            # Add numbers to cached suggestions
            for i, suggestion in enumerate(cached_suggestions[:max_suggestions]):
                suggestion.number = i + 1
            return cached_suggestions[:max_suggestions]
        
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
            for word in list(self.trie.word_frequencies.keys())[:50]:
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
        
        # 4. Rank suggestions
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
        return HindiDistance._hindi_levenshtein(s1, s2)
    
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
autocorrect = HindiBrailleAutoCorrect()

# Sample Hindi words (100 common words)
SAMPLE_WORDS = [
    # Common nouns
    "पानी", "आदमी", "औरत", "बच्चा", "घर", "गाड़ी", "किताब", "स्कूल", "दोस्त", "परिवार",
    "माता", "पिता", "भाई", "बहन", "बेटा", "बेटी", "नाम", "काम", "समय", "दिन",
    "रात", "सुबह", "शाम", "साल", "महीना", "हफ्ता", "पैसा", "खाना", "चाय", "दूध",
    
    # Common verbs
    "आना", "जाना", "करना", "होना", "देना", "लेना", "खाना", "पीना", "सोना", "उठना",
    "बैठना", "खड़ा", "चलना", "दौड़ना", "पढ़ना", "लिखना", "बोलना", "सुनना", "देखना", "समझना",
    
    # Common adjectives
    "अच्छा", "बुरा", "बड़ा", "छोटा", "नया", "पुराना", "लंबा", "छोटा", "मोटा", "पतला",
    "सुंदर", "सफेद", "काला", "लाल", "नीला", "हरा", "पीला", "गर्म", "ठंडा", "तेज",
    
    # Common pronouns and particles
    "मैं", "तुम", "वह", "हम", "आप", "यह", "वह", "कौन", "क्या", "कब",
    "कहाँ", "कैसे", "क्यों", "और", "या", "लेकिन", "अगर", "तो", "भी", "नहीं",
    
    # Numbers
    "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ", "दस",
    
    # Common words
    "हाँ", "ना", "कृपया", "धन्यवाद", "माफ", "हैलो", "नमस्ते", "अलविदा", "शुभ", "रात्रि"
]

def initialize_autocorrect():
    """Initialize the Hindi autocorrect system with sample words"""
    print("Initializing Hindi Braille AutoCorrect with sample words...")
    
    # Create frequency distribution (higher frequency for more common words)
    frequencies = {}
    for i, word in enumerate(SAMPLE_WORDS):
        # More common words get higher frequency
        frequencies[word] = len(SAMPLE_WORDS) - i
    
    autocorrect.load_dictionary(SAMPLE_WORDS, frequencies)
    print(f"Loaded Hindi dictionary with {len(SAMPLE_WORDS)} words")

# Initialize on startup
initialize_autocorrect()

            
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
        
        hindi_letter = BrailleInputProcessor.convert_braille_to_text(braille_input)
        
        return jsonify({
            'braille_input': braille_input,
            'hindi_letter': hindi_letter,
            'valid': hindi_letter != '?'
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
        hindi_word = BrailleInputProcessor.convert_braille_sequence_to_word(braille_sequence)
        
        # Get suggestions
        suggestions = autocorrect.suggest_from_braille(braille_sequence, max_suggestions)
        
        # Convert suggestions to dictionaries for JSON serialization
        suggestions_data = [asdict(s) for s in suggestions]
        
        return jsonify({
            'braille_sequence': braille_sequence,
            'converted_word': hindi_word,
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
        hindi_word = BrailleInputProcessor.convert_braille_sequence_to_word(braille_sequence)
        
        # Learn from the selection
        autocorrect.learn(hindi_word, selected_word)
        
        return jsonify({
            'message': f'Selected suggestion #{suggestion_number}: "{selected_word}"',
            'braille_sequence': braille_sequence,
            'converted_word': hindi_word,
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