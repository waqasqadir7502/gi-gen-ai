import React from 'react';
import './ChatWidget.css';

const MessageList = ({ messages, isLoading, error }) => {
  const renderSources = (sources) => {
    if (!sources || sources.length === 0) return null;

    return (
      <div className="message-sources">
        <strong>Sources:</strong>
        <ul>
          {sources.map((source, index) => (
            <li key={index}>
              <a href={`#${source}`} target="_blank" rel="noopener noreferrer">
                {source}
              </a>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  const renderMessageContent = (message) => {
    if (message.sender === 'ai' && message.isError) {
      return (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          <span>{message.text}</span>
        </div>
      );
    }

    // Simple markdown-like formatting for links and bold
    let formattedText = message.text
      .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    return (
      <div>
        <div
          className="message-text"
          dangerouslySetInnerHTML={{ __html: formattedText }}
        />
        {message.sources && renderSources(message.sources)}
      </div>
    );
  };

  return (
    <div className="message-list">
      {messages.length === 0 ? (
        <div className="welcome-message">
          <h3>Hello! 👋</h3>
          <p>Ask me anything about the Physical AI Book, and I'll help you find relevant information.</p>
        </div>
      ) : (
        messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.sender}-message`}
            role="log"
            aria-live="polite"
          >
            <div className="message-avatar">
              {message.sender === 'user' ? '👤' : '🤖'}
            </div>
            <div className="message-content">
              {renderMessageContent(message)}
              <div className="message-timestamp">
                {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          </div>
        ))
      )}

      {isLoading && (
        <div className="message ai-message">
          <div className="message-avatar">🤖</div>
          <div className="message-content">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="message ai-message">
          <div className="message-avatar">🤖</div>
          <div className="message-content">
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              <span>Error: {error}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MessageList;