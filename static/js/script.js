        let currentSentence = [];
        let currentSuggestions = [];
        let isLoading = false;

        // Update sentence display
        function updateSentenceDisplay() {
            const sentenceText = document.getElementById('sentenceText');
            const wordCount = document.getElementById('wordCount');
            
            if (currentSentence.length === 0) {
                sentenceText.textContent = 'Start typing your sentence...';
                sentenceText.style.color = '#999';
            } else {
                sentenceText.textContent = currentSentence.join(' ');
                sentenceText.style.color = '#333';
            }
            
            wordCount.textContent = `Words: ${currentSentence.length}`;
        }

        // Update status bar
        function updateStatus(message, isError = false) {
            const statusBar = document.getElementById('statusBar');
            statusBar.textContent = message;
            statusBar.style.background = isError ? '#f44336' : '#333';
        }

        // Convert Braille input to English in real-time
        async function convertBrailleInput() {
            const brailleInput = document.getElementById('brailleInput').value.trim();
            const conversionDisplay = document.getElementById('conversionDisplay');
            
            if (!brailleInput) {
                conversionDisplay.textContent = 'Converted word will appear here...';
                conversionDisplay.style.color = '#999';
                return;
            }

            try {
                const response = await fetch('/api/suggest-word', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        braille_sequence: brailleInput,
                        max_suggestions: 5
                    })
                });

                const data = await response.json();
                
                if (response.ok) {
                    conversionDisplay.textContent = `"${brailleInput}" → "${data.converted_word}"`;
                    conversionDisplay.style.color = data.converted_word.includes('?') ? '#d32f2f' : '#333';
                } else {
                    conversionDisplay.textContent = `Error: ${data.error}`;
                    conversionDisplay.style.color = '#d32f2f';
                }
            } catch (error) {
                conversionDisplay.textContent = `Network error: ${error.message}`;
                conversionDisplay.style.color = '#d32f2f';
            }
        }

        // Get suggestions for current input
        async function getSuggestions() {
            if (isLoading) return;
            
            const brailleInput = document.getElementById('brailleInput').value.trim();
            
            if (!brailleInput) {
                updateStatus('Please enter Braille input first', true);
                return;
            }

            isLoading = true;
            updateStatus('Getting suggestions...');

            try {
                const response = await fetch('/api/suggest-word', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        braille_sequence: brailleInput,
                        max_suggestions: 5
                    })
                });

                const data = await response.json();
                
                if (response.ok) {
                    displaySuggestions(data.suggestions, brailleInput);
                    updateStatus(`Found ${data.count} suggestions for "${data.converted_word}"`);
                } else {
                    updateStatus(`Error: ${data.error}`, true);
                }
            } catch (error) {
                updateStatus(`Network error: ${error.message}`, true);
            } finally {
                isLoading = false;
            }
        }

        // Display suggestions
        function displaySuggestions(suggestions, brailleInput) {
            currentSuggestions = suggestions;
            const container = document.getElementById('suggestionsContainer');
            const list = document.getElementById('suggestionsList');
            
            if (suggestions.length === 0) {
                container.style.display = 'none';
                return;
            }

            list.innerHTML = '';
            
            suggestions.forEach((suggestion, index) => {
                const item = document.createElement('div');
                item.className = 'suggestion-item';
                item.onclick = () => selectSuggestion(suggestion, brailleInput);
                
                item.innerHTML = `
                    <div class="suggestion-number">${suggestion.number}</div>
                    <div class="suggestion-word">${suggestion.word}</div>
                    <div class="suggestion-meta">
                        ${(suggestion.confidence * 100).toFixed(0)}% | ${suggestion.source}
                    </div>
                `;
                
                list.appendChild(item);
            });
            
            container.style.display = 'block';
        }

        // Select a suggestion
        async function selectSuggestion(suggestion, brailleInput) {
            try {
                // Record the selection for learning
                await fetch('/api/select-suggestion', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        braille_sequence: brailleInput,
                        selected_word: suggestion.word,
                        suggestion_number: suggestion.number
                    })
                });

                // Add word to sentence
                currentSentence.push(suggestion.word);
                updateSentenceDisplay();
                
                // Clear input and suggestions
                clearCurrentInput();
                
                updateStatus(`Added "${suggestion.word}" to sentence`);
            } catch (error) {
                updateStatus(`Error selecting suggestion: ${error.message}`, true);
            }
        }

        // Add current converted word to sentence
        async function addCurrentWord() {
            const brailleInput = document.getElementById('brailleInput').value.trim();
            
            if (!brailleInput) {
                updateStatus('Please enter Braille input first', true);
                return;
            }

            try {
                const response = await fetch('/api/suggest-word', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        braille_sequence: brailleInput,
                        max_suggestions: 1
                    })
                });

                const data = await response.json();
                
                if (response.ok && !data.converted_word.includes('?')) {
                    currentSentence.push(data.converted_word);
                    updateSentenceDisplay();
                    clearCurrentInput();
                    updateStatus(`Added "${data.converted_word}" to sentence`);
                } else {
                    updateStatus('Cannot add invalid word to sentence', true);
                }
            } catch (error) {
                updateStatus(`Error: ${error.message}`, true);
            }
        }

        // Clear current input
        function clearCurrentInput() {
            document.getElementById('brailleInput').value = '';
            document.getElementById('conversionDisplay').textContent = 'Converted word will appear here...';
            document.getElementById('conversionDisplay').style.color = '#999';
            document.getElementById('suggestionsContainer').style.display = 'none';
            currentSuggestions = [];
            updateStatus('Input cleared');
        }

        // Clear entire sentence
        function clearSentence() {
            if (currentSentence.length > 0 && confirm('Are you sure you want to clear the entire sentence?')) {
                currentSentence = [];
                updateSentenceDisplay();
                updateStatus('Sentence cleared');
            }
        }

        // Event listeners
        document.getElementById('brailleInput').addEventListener('input', convertBrailleInput);

        document.getElementById('brailleInput').addEventListener('keydown', function(event) {
            // Enter key - get suggestions
            if (event.key === 'Enter' && !event.ctrlKey) {
                event.preventDefault();
                getSuggestions();
            }
            // Ctrl+Enter - add current word
            else if (event.key === 'Enter' && event.ctrlKey) {
                event.preventDefault();
                addCurrentWord();
            }
            // Escape - clear input
            else if (event.key === 'Escape') {
                event.preventDefault();
                clearCurrentInput();
            }
            // Number keys 1-5 - select suggestion
            else if (event.key >= '1' && event.key <= '5' && event.ctrlKey) {
                const suggestionIndex = parseInt(event.key) - 1;
                if (currentSuggestions[suggestionIndex]) {
                    event.preventDefault();
                    selectSuggestion(currentSuggestions[suggestionIndex], this.value.trim());
                }
            }
        });

        // Initialize
        updateSentenceDisplay();
        updateStatus('Ready - Enter Braille input using d,w,k,o,q,p keys');

        // Focus on input field
        document.getElementById('brailleInput').focus();
