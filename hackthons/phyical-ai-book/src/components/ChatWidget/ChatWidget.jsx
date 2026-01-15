import React, { useState } from 'react';
import './ChatWidget.css';
import ChatWindow from './ChatWindow';

const ChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);

  const toggleChat = () => {
    if (!isOpen) {
      setIsOpen(true);
      setIsMinimized(false);
    } else {
      setIsMinimized(!isMinimized);
    }
  };

  const closeChat = () => {
    setIsOpen(false);
    setIsMinimized(false);
  };

  const minimizeChat = () => {
    setIsMinimized(true);
  };

  return (
    <div className="chat-widget">
      {isOpen && !isMinimized ? (
        <ChatWindow onClose={closeChat} onMinimize={minimizeChat} />
      ) : isOpen && isMinimized ? (
        <div className="chat-minimized" onClick={() => setIsMinimized(false)}>
          <span>💬 AI Assistant</span>
        </div>
      ) : (
        <button
          className="chat-fab"
          onClick={toggleChat}
          aria-label="Open chat"
          title="AI Assistant"
        >
          💬
        </button>
      )}
    </div>
  );
};

export default ChatWidget;