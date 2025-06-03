from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import importlib.util
import sys
import os

# Import the individual language apps
try:
    # Import English app
    spec_english = importlib.util.spec_from_file_location("app_english", "app_english.py")
    app_english_module = importlib.util.module_from_spec(spec_english)
    sys.modules["app_english"] = app_english_module
    spec_english.loader.exec_module(app_english_module)
    
    # Import Hindi app
    spec_hindi = importlib.util.spec_from_file_location("app_hindi", "app_hindi.py")
    app_hindi_module = importlib.util.module_from_spec(spec_hindi)
    sys.modules["app_hindi"] = app_hindi_module
    spec_hindi.loader.exec_module(app_hindi_module)
    
    print("Successfully imported both language modules")
    
except Exception as e:
    print(f"Error importing language modules: {e}")
    print("Please ensure app_english.py and app_hindi.py exist in the same directory")
    sys.exit(1)

# Create main Flask app
main_app = Flask(__name__)
main_app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
CORS(main_app)

# Language configuration
SUPPORTED_LANGUAGES = {
    'english': {
        'name': 'English',
        'code': 'en',
        'app_module': app_english_module,
        'display_name': 'English'
    },
    'hindi': {
        'name': 'Hindi',
        'code': 'hi', 
        'app_module': app_hindi_module,
        'display_name': 'हिंदी'
    }
}

DEFAULT_LANGUAGE = 'english'

def get_current_language():
    """Get current language from session or default"""
    return session.get('language', DEFAULT_LANGUAGE)

def get_current_app_module():
    """Get the Flask app module for current language"""
    language = get_current_language()
    return SUPPORTED_LANGUAGES[language]['app_module']

def get_autocorrect_instance():
    """Get the autocorrect instance for current language"""
    app_module = get_current_app_module()
    return getattr(app_module, 'autocorrect', None)

# Main routes
@main_app.route('/')
def index():
    """Main index page with language selection"""
    current_language = get_current_language()
    return render_template('index.html', 
                         current_language=current_language,
                         languages=SUPPORTED_LANGUAGES)

@main_app.route('/set-language/<language>')
def set_language(language):
    """Set the current language"""
    if language in SUPPORTED_LANGUAGES:
        session['language'] = language
        return jsonify({
            'success': True, 
            'message': f'Language set to {SUPPORTED_LANGUAGES[language]["name"]}',
            'language': language,
            'display_name': SUPPORTED_LANGUAGES[language]['display_name']
        })
    else:
        return jsonify({
            'success': False, 
            'message': 'Unsupported language'
        }), 400

@main_app.route('/api/current-language')
def get_current_language_api():
    """API endpoint to get current language"""
    current_language = get_current_language()
    return jsonify({
        'language': current_language,
        'display_name': SUPPORTED_LANGUAGES[current_language]['display_name'],
        'supported_languages': {
            lang: {
                'name': info['name'],
                'display_name': info['display_name'],
                'code': info['code']
            } for lang, info in SUPPORTED_LANGUAGES.items()
        }
    })

# Proxy routes - Forward requests to appropriate language app
@main_app.route('/api/convert-braille', methods=['POST'])
def convert_braille():
    """Convert Braille input - proxy to language-specific handler"""
    try:
        app_module = get_current_app_module()
        
        # Call the convert_braille function from the appropriate module
        if hasattr(app_module, 'convert_braille'):
            # If it's a function, call it directly
            return app_module.convert_braille()
        else:
            # If it's part of the Flask app, we need to simulate the request
            data = request.get_json()
            braille_input = data.get('input', '').strip()
            
            if not braille_input:
                return jsonify({'error': 'No input provided'}), 400
            
            # Use the BrailleInputProcessor from the current language module
            processor_class = getattr(app_module, 'BrailleInputProcessor', None)
            if processor_class:
                english_letter = processor_class.convert_braille_to_text(braille_input)
                return jsonify({
                    'braille_input': braille_input,
                    'english_letter': english_letter,
                    'valid': english_letter != '?'
                })
            else:
                return jsonify({'error': 'BrailleInputProcessor not found'}), 500
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_app.route('/api/suggest-word', methods=['POST'])
def suggest_word():
    """Get word suggestions - proxy to language-specific handler"""
    try:
        data = request.get_json()
        braille_sequence = data.get('braille_sequence', '').strip()
        max_suggestions = data.get('max_suggestions', 5)
        
        if not braille_sequence:
            return jsonify({'error': 'No braille sequence provided'}), 400
        
        app_module = get_current_app_module()
        autocorrect_instance = getattr(app_module, 'autocorrect', None)
        
        if not autocorrect_instance:
            return jsonify({'error': 'Autocorrect system not available'}), 500
        
        # Convert Braille sequence to target language word
        processor_class = getattr(app_module, 'BrailleInputProcessor', None)
        if processor_class:
            converted_word = processor_class.convert_braille_sequence_to_word(braille_sequence)
        else:
            converted_word = braille_sequence
        
        # Get suggestions
        suggestions = autocorrect_instance.suggest_from_braille(braille_sequence, max_suggestions)
        
        # Convert suggestions to dictionaries for JSON serialization
        from dataclasses import asdict
        suggestions_data = [asdict(s) for s in suggestions]
        
        return jsonify({
            'braille_sequence': braille_sequence,
            'converted_word': converted_word,
            'suggestions': suggestions_data,
            'count': len(suggestions_data),
            'language': get_current_language()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_app.route('/api/select-suggestion', methods=['POST'])
def select_suggestion():
    """Record suggestion selection - proxy to language-specific handler"""
    try:
        data = request.get_json()
        braille_sequence = data.get('braille_sequence', '').strip()
        selected_word = data.get('selected_word', '').strip()
        suggestion_number = data.get('suggestion_number', 0)
        
        if not braille_sequence or not selected_word:
            return jsonify({'error': 'Both braille_sequence and selected_word are required'}), 400
        
        app_module = get_current_app_module()
        autocorrect_instance = getattr(app_module, 'autocorrect', None)
        
        if not autocorrect_instance:
            return jsonify({'error': 'Autocorrect system not available'}), 500
        
        # Convert Braille to target language for learning
        processor_class = getattr(app_module, 'BrailleInputProcessor', None)
        if processor_class:
            converted_word = processor_class.convert_braille_sequence_to_word(braille_sequence)
        else:
            converted_word = braille_sequence
        
        # Learn from the selection
        autocorrect_instance.learn(converted_word, selected_word)
        
        return jsonify({
            'message': f'Selected suggestion #{suggestion_number}: "{selected_word}"',
            'braille_sequence': braille_sequence,
            'converted_word': converted_word,
            'selected_word': selected_word,
            'suggestion_number': suggestion_number,
            'language': get_current_language()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_app.route('/api/build-sentence', methods=['POST'])
def build_sentence():
    """Build sentence - proxy to language-specific handler"""
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
            'word_count': len(updated_sentence),
            'language': get_current_language()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_app.route('/api/stats')
def get_stats():
    """Get system statistics - proxy to language-specific handler"""
    try:
        autocorrect_instance = get_autocorrect_instance()
        if not autocorrect_instance:
            return jsonify({'error': 'Autocorrect system not available'}), 500
        
        stats = autocorrect_instance.get_stats()
        stats['current_language'] = get_current_language()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@main_app.route('/api/health')
def health_check():
    """Health check endpoint"""
    current_language = get_current_language()
    autocorrect_instance = get_autocorrect_instance()
    
    return jsonify({
        'status': 'healthy',
        'current_language': current_language,
        'autocorrect_available': autocorrect_instance is not None,
        'supported_languages': list(SUPPORTED_LANGUAGES.keys()),
        'session_id': session.get('session_id', 'new')
    })

# Error handlers
@main_app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@main_app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize session
@main_app.before_request
def before_request():
    """Initialize session before each request"""
    if 'session_id' not in session:
        import uuid
        session['session_id'] = str(uuid.uuid4())
    
    if 'language' not in session:
        session['language'] = DEFAULT_LANGUAGE

if __name__ == '__main__':
    print("Starting multi-language Braille autocorrect system...")
    print(f"Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")
    print(f"Default language: {DEFAULT_LANGUAGE}")
    
    # Verify that both language modules loaded successfully
    for lang, config in SUPPORTED_LANGUAGES.items():
        module = config['app_module']
        autocorrect_attr = getattr(module, 'autocorrect', None)
        if autocorrect_attr:
            print(f"✓ {config['name']} ({config['display_name']}) module loaded successfully")
        else:
            print(f"⚠ {config['name']} module loaded but autocorrect instance not found")
    
    main_app.run()
