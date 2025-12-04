import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import TypingIndicator from './TypingIndicator';

// Format text with proper line breaks and styling
const formatMessage = (text) => {
  // Split by numbered items (1. 2. 3. etc) or newlines
  const lines = text.split(/(?=\d+\.\s)|(?:\n)/g).filter(line => line.trim());
  
  return lines.map((line, index) => {
    const trimmedLine = line.trim();
    // Check if it's a numbered item
    if (/^\d+\.\s/.test(trimmedLine)) {
      return (
        <div key={index} className="message-list-item">
          {trimmedLine}
        </div>
      );
    }
    // Check if it's a section header like "Related questions:"
    if (trimmedLine.toLowerCase().includes('related questions')) {
      return (
        <div key={index} className="message-section-header">
          {trimmedLine}
        </div>
      );
    }
    // Regular text
    return (
      <div key={index} className="message-paragraph">
        {trimmedLine}
      </div>
    );
  });
};

const Chat = () => {
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([
    {
      text: "Hi there! 👋 I'm your X (Twitter) Support Assistant. How can I help you today?",
      sender: 'bot',
    }
  ]);
  
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      text: inputValue,
      sender: 'user',
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await axios.post('/chat', { user_message: inputValue });
      const botMessage = {
        text: response.data.response,
        sender: 'bot',
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        text: "I'm sorry, something went wrong. Please try again.",
        sender: 'bot',
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-wrapper">
      {/* Header */}
      <div className="chat-header">
        <div className="header-avatar">
          <svg viewBox="0 0 24 24" width="28" height="28" fill="#1d9bf0">
            <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
          </svg>
        </div>
        <div className="header-info">
          <div className="header-title">X Support Assistant</div>
          <div className="header-status">
            <span className="status-dot"></span>
            Always here to help
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="messages-container">
        <div className="welcome-message">
          <strong>Welcome to X Support</strong>
          Ask me anything about your account, settings, or features.
        </div>
        
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender}`}>
            <div className="message-avatar">
              {message.sender === 'bot' ? (
                <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
                  <path d="M12 12c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm0 2c-3.33 0-10 1.67-10 5v2h20v-2c0-3.33-6.67-5-10-5z"/>
                </svg>
              )}
            </div>
            <div className="message-content">
              {message.sender === 'bot' ? formatMessage(message.text) : message.text}
            </div>
          </div>
        ))}
        
        {isLoading && <TypingIndicator />}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="input-container">
        <input
          type="text"
          className="message-input"
          placeholder="Type your message..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isLoading}
        />
        <button 
          className="send-button" 
          onClick={handleSend}
          disabled={isLoading || !inputValue.trim()}
        >
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default Chat;
