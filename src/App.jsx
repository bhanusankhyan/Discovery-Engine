import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, AlertCircle, RefreshCw } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const textareaRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom on new messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Handle textarea height adjustment based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.max(Math.min(textareaRef.current.scrollHeight, 200), 72)}px`;
    }
  }, [inputValue]);

  const handleSend = async (textToSend) => {
    const text = textToSend || inputValue.trim();
    if (!text) return;

    // Clear input
    if (!textToSend) {
      setInputValue('');
    }
    setError(null);

    // Create user message
    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // POST request to backend
      const response = await fetch('https://discovery-engine.onrender.com/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: text }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const contentType = response.headers.get('content-type');
      let botResponseText = '';

      if (contentType && contentType.includes('application/json')) {
        const data = await response.json();
        // Check standard keys for response text
        botResponseText = data.response || data.reply || data.message || data.output || JSON.stringify(data);
      } else {
        botResponseText = await response.text();
      }

      // Create bot message
      const botMessage = {
        id: (Date.now() + 1).toString(),
        role: 'bot',
        content: botResponseText,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      console.error('Error fetching query response:', err);
      setError({
        message: 'Could not connect to the backend server. Please verify it is running on http://localhost:8000/query',
        retryQuery: text
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleRetry = () => {
    if (error && error.retryQuery) {
      const query = error.retryQuery;
      // Remove the last user message to avoid duplication when retrying
      setMessages(prev => prev.slice(0, -1));
      handleSend(query);
    }
  };

  const suggestions = [
    "Design a minimalist workspace layout",
    "How do I set up a local CORS proxy?",
    "Summarize key features of React 19",
    "Explain quantum entanglement simply"
  ];

  return (
    <div className="app-container">
      <main className="chat-window">
        {messages.length === 0 ? (
          <div className="welcome-container">
            <h1 className="welcome-quote">"Simplified Q-commerce"</h1>
          </div>
        ) : (
          <div className="messages-container">
            <AnimatePresence initial={false}>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35, ease: 'easeOut' }}
                  className={`message-row ${message.role}`}
                >
                  <div className={`avatar ${message.role}`}>
                    {message.role === 'user' ? 'U' : 'AI'}
                  </div>
                  <div className="message-content">
                    <div className="bubble">
                      {message.content}
                    </div>
                    <span className="message-time">{message.timestamp}</span>
                  </div>
                </motion.div>
              ))}

              {isLoading && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="message-row bot"
                >
                  <div className="avatar bot">AI</div>
                  <div className="message-content">
                    <div className="bubble typing-bubble">
                      <div className="dot"></div>
                      <div className="dot"></div>
                      <div className="dot"></div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {error && (
              <div className="error-banner">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <AlertCircle size={18} />
                  <span>{error.message}</span>
                </div>
                <button onClick={handleRetry} className="error-retry-btn">
                  <RefreshCw size={14} style={{ marginRight: '4px', display: 'inline' }} />
                  Retry
                </button>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}

        <div className="input-area-fixed">
          <div className="input-wrapper">
            <textarea
              ref={textareaRef}
              rows={3}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Your thoughts..."
              className="chat-textarea"
              disabled={isLoading}
            />
            <button
              onClick={() => handleSend()}
              disabled={!inputValue.trim() || isLoading}
              className="send-button"
              aria-label="Send message"
            >
              <Send size={16} />
            </button>
          </div>
          <div className="footer-info">
            Assistant can make mistakes. Verify important info.
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
