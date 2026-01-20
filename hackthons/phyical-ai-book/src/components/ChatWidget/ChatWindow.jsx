import React, { useState, useRef, useEffect } from 'react';
import MessageList from './MessageList';
import InputArea from './InputArea';
import './ChatWidget.css';

const ChatWindow = ({ onClose, onMinimize }) => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (question, context = null) => {
    if (!question.trim()) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      text: question,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      // Retry mechanism
      const maxRetries = 3;
      let response = null;
      let lastError = null;

      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
          // Call the backend API with timeout
          // Use appropriate API endpoint based on environment
          // During local development, we'll use http://localhost:8000
          // For Vercel deployment, use the deployed backend URL

          const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';
          const apiBaseUrl = isLocalhost ? 'http://localhost:8000' : 'https://physical-ai-book-lilac.vercel.app'; // Deployed backend URL

          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

          const apiUrl = apiBaseUrl ? `${apiBaseUrl}/api/chat` : '/api/chat';

          response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-API-Key': localStorage.getItem('backendApiKey') || '63f4945d921d599f27ae4fdf5bada3f1', // Use the key from .env file
            },
            body: JSON.stringify({
              question: question,
              context: context,
              session_id: null // Future extension
            }),
            signal: controller.signal
          });

          clearTimeout(timeoutId);

          if (response.ok) {
            break; // Success, exit retry loop
          } else if (response.status === 429) {
            // Rate limited - wait before retrying
            const retryAfter = parseInt(response.headers.get('Retry-After')) || 60;
            console.warn(`Rate limited. Waiting ${retryAfter} seconds before retrying...`);
            await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
          }
        } catch (err) {
          lastError = err;
          if (err.name === 'AbortError') {
            throw new Error('Request timed out after 30 seconds');
          }

          if (attempt === maxRetries) {
            throw err; // Last attempt, re-throw the error
          }

          // Wait before retrying (exponential backoff)
          const waitTime = Math.pow(2, attempt) * 1000; // 2, 4, 8 seconds
          console.log(`Attempt ${attempt} failed, retrying in ${waitTime}ms...`);
          await new Promise(resolve => setTimeout(resolve, waitTime));
        }
      }

      if (!response || !response.ok) {
        throw new Error(lastError?.message || `API request failed with status ${response?.status}`);
      }

      const data = await response.json();

      // Add AI response
      const aiMessage = {
        id: Date.now() + 1,
        text: data.answer,
        sender: 'ai',
        sources: data.sources || [],
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (err) {
      console.error('Error sending message:', err);

      // Format error message for display
      let errorMessageText = 'Sorry, I encountered an error processing your request. Please try again.';
      if (err.message.includes('timeout')) {
        errorMessageText = 'The request timed out. Please try again with a shorter question.';
      } else if (err.message.includes('429')) {
        errorMessageText = 'Too many requests. Please wait a moment before asking another question.';
      }

      setError(err.message);
      const errorMessage = {
        id: Date.now() + 1,
        text: errorMessageText,
        sender: 'ai',
        isError: true,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-window">
      <div className="chat-header">
        <div className="chat-title">AI Assistant</div>
        <div className="chat-controls">
          <button
            className="control-button minimize-btn"
            onClick={onMinimize}
            aria-label="Minimize chat"
          >
            −
          </button>
          <button
            className="control-button close-btn"
            onClick={onClose}
            aria-label="Close chat"
          >
            ×
          </button>
        </div>
      </div>

      <MessageList messages={messages} isLoading={isLoading} error={error} />

      <InputArea
        onSend={handleSendMessage}
        disabled={isLoading}
      />

      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatWindow;