import React, { useState, useRef, useEffect } from 'react';
import textSelectionManager from './textSelection';
import './ChatWidget.css';

const InputArea = ({ onSend, disabled }) => {
  const [inputValue, setInputValue] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedContext, setSelectedContext] = useState(null);
  const [showContextPreview, setShowContextPreview] = useState(false);
  const textareaRef = useRef(null);

  // Register for text selection changes
  useEffect(() => {
    const handleSelectionChange = (selectedText) => {
      if (selectedText && textSelectionManager.isValidSelection(selectedText)) {
        const context = textSelectionManager.getFormattedContext();
        if (context) {
          setSelectedContext(context);
          setShowContextPreview(true);
        }
      } else {
        setSelectedContext(null);
        setShowContextPreview(false);
      }
    };

    textSelectionManager.onSelection(handleSelectionChange);

    // Cleanup listener
    return () => {
      textSelectionManager.onSelection(null);
    };
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim() && !disabled) {
      // Pass the selected context if available
      onSend(inputValue.trim(), selectedContext ? selectedContext.selectedText : null);
      setInputValue('');
      setIsExpanded(false);
      setSelectedContext(null);
      setShowContextPreview(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    } else if (e.key === 'Escape') {
      setInputValue('');
      setIsExpanded(false);
      if (showContextPreview) {
        setShowContextPreview(false);
      }
    } else if (e.ctrlKey && e.key === 'k') {
      // Ctrl+K to include selected context
      e.preventDefault();
      if (selectedContext) {
        setInputValue(prev => prev + ' [Selected Context: ' + selectedContext.selectedText.substring(0, 100) + '...]');
      }
    }
  };

  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;

      // Set expanded state based on height
      setIsExpanded(textareaRef.current.scrollHeight > 60);
    }
  };

  const removeSelectedContext = () => {
    setSelectedContext(null);
    setShowContextPreview(false);
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [inputValue]);

  return (
    <form className="input-area" onSubmit={handleSubmit}>
      {/* Context Preview */}
      {showContextPreview && selectedContext && (
        <div className="context-preview">
          <div className="context-header">
            <span>📎 Selected Context:</span>
            <button
              type="button"
              className="remove-context-btn"
              onClick={removeSelectedContext}
              aria-label="Remove selected context"
            >
              ×
            </button>
          </div>
          <div className="context-content">
            {selectedContext.selectedText.length > 100
              ? selectedContext.selectedText.substring(0, 100) + '...'
              : selectedContext.selectedText}
          </div>
          <div className="context-actions">
            <button
              type="button"
              className="include-context-btn"
              onClick={() => {
                setInputValue(prev => prev + ' [Context: ' + selectedContext.selectedText + ']');
                setShowContextPreview(false);
              }}
            >
              Include in Question
            </button>
          </div>
        </div>
      )}

      <div className="input-container">
        <textarea
          ref={textareaRef}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about the Physical AI Book... (selected text will appear above)"
          disabled={disabled}
          aria-label="Type your question"
          rows={1}
          className={`input-textarea ${isExpanded ? 'expanded' : ''} ${selectedContext ? 'has-context' : ''}`}
        />
        <button
          type="submit"
          disabled={!inputValue.trim() || disabled}
          className="send-button"
          aria-label="Send message"
        >
          {disabled ? (
            <span className="loading-spinner">⏳</span>
          ) : (
            <span>➤</span>
          )}
        </button>
      </div>
      <div className="input-hints">
        <small>Enter: Send • Shift+Enter: New line • Esc: Clear • Ctrl+K: Include context</small>
      </div>
    </form>
  );
};

export default InputArea;