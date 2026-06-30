# 4.5 前端 AI 应用开发

> 将大模型能力集成到前端应用，构建流畅的 AI 交互体验。

## 学习目标

学完本节后，你将能够：

- [ ] 实现流式对话 UI
- [ ] 处理 Markdown 和代码高亮
- [ ] 管理对话状态和历史
- [ ] 优化前端 AI 体验

---

## 4.5.1 React + AI 基础架构

### 项目结构

```
frontend-ai-app/
├── src/
│   ├── components/
│   │   ├── ChatInput.tsx      # 输入框
│   │   ├── MessageList.tsx    # 消息列表
│   │   ├── MessageBubble.tsx  # 消息气泡
│   │   ├── MarkdownRenderer.tsx
│   │   └── CodeBlock.tsx
│   ├── hooks/
│   │   ├── useChat.ts         # 对话逻辑
│   │   └── useStream.ts       # 流式处理
│   ├── stores/
│   │   └── chatStore.ts       # 状态管理
│   ├── services/
│   │   └── api.ts             # API 调用
│   └── types/
│       └── chat.ts            # 类型定义
```

---

## 4.5.2 核心组件实现

### 聊天输入组件

```tsx
// examples/4-5-frontend/src/components/ChatInput.tsx
import React, { useState, useRef, useEffect } from 'react';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  disabled = false,
  placeholder = '输入消息...'
}) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // 自动调整高度
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = 
        `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSend(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="chat-input-form">
      <textarea
        ref={textareaRef}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        rows={1}
        className="chat-input-textarea"
      />
      <button 
        type="submit" 
        disabled={!input.trim() || disabled}
        className="send-button"
      >
        <SendIcon />
      </button>
    </form>
  );
};
```

### 消息列表组件

```tsx
// examples/4-5-frontend/src/components/MessageList.tsx
import React, { useRef, useEffect } from 'react';
import { MessageBubble } from './MessageBubble';
import { MarkdownRenderer } from './MarkdownRenderer';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

interface MessageListProps {
  messages: Message[];
  streamingContent?: string;
  isLoading?: boolean;
}

export const MessageList: React.FC<MessageListProps> = ({
  messages,
  streamingContent,
  isLoading = false
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  return (
    <div className="message-list">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      
      {/* 流式输出 */}
      {streamingContent && (
        <div className="message assistant streaming">
          <MarkdownRenderer content={streamingContent} />
        </div>
      )}
      
      {/* 加载状态 */}
      {isLoading && !streamingContent && (
        <div className="message assistant loading">
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      )}
      
      <div ref={messagesEndRef} />
    </div>
  );
};
```

### Markdown 渲染器

```tsx
// examples/4-5-frontend/src/components/MarkdownRenderer.tsx
import React from 'react';
import ReactMarkdown from 'react-markdown';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-bash';
import 'prismjs/themes/prism-tomorrow.css';

interface MarkdownRendererProps {
  content: string;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content }) => {
  return (
    <ReactMarkdown
      children={content}
      components={{
        // 代码块
        code({ node, inline, className, children, ...props }: any) {
          const match = /language-(\w+)/.exec(className || '');
          
          if (!inline && match) {
            return (
              <div className="code-block-wrapper">
                <div className="code-block-header">
                  <span className="language-label">{match[1]}</span>
                  <button 
                    onClick={() => copyCode(children)}
                    className="copy-button"
                  >
                    复制
                  </button>
                </div>
                <pre className={`language-${match[1]}`}>
                  <code className={className} {...props}>
                    {String(children).replace(/\n$/, '')}
                  </code>
                </pre>
              </div>
            );
          }
          
          return <code className={className}>{children}</code>;
        },
        
        // 表格
        table({ children }) {
          return (
            <div className="table-wrapper">
              <table>{children}</table>
            </div>
          );
        },
        
        // 数学公式（使用 KaTeX）
        span({ children }: any) {
          // 检测是否为 LaTeX 公式
          if (typeof children === 'string' && children.startsWith('\\(')) {
            return <InlineMath math={children} />;
          }
          return <span>{children}</span>;
        }
      }}
    />
  );
};

// 代码复制功能
const copyCode = (code: React.ReactNode) => {
  const text = String(code).replace(/\n$/, '');
  navigator.clipboard.writeText(text);
};
```

---

## 4.5.3 流式响应处理

```tsx
// examples/4-5-frontend/src/hooks/useStream.ts
import { useState, useCallback } from 'react';

interface UseStreamOptions {
  onError?: (error: Error) => void;
  onComplete?: () => void;
}

export function useStream(options: UseStreamOptions = {}) {
  const [content, setContent] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // 流式读取
  const stream = useCallback(async (
    url: string,
    body: object
  ) => {
    setIsLoading(true);
    setContent('');
    setError(null);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        // 解析 SSE 格式
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // 保留不完整行

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data);
              const token = parsed.choices?.[0]?.delta?.content || '';
              if (token) {
                setContent(prev => prev + token);
              }
            } catch {
              // 忽略解析错误
            }
          }
        }
      }

      options.onComplete?.();
    } catch (err) {
      const error = err as Error;
      setError(error);
      options.onError?.(error);
    } finally {
      setIsLoading(false);
    }
  }, [options]);

  return { content, isLoading, error, stream };
}
```

### 对话 Hook

```tsx
// examples/4-5-frontend/src/hooks/useChat.ts
import { useState, useCallback } from 'react';
import { useStream } from './useStream';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

export function useChat(apiUrl: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const { content: streamingContent, stream, isLoading } = useStream({
    onComplete: () => {
      // 流式完成后添加到历史
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'assistant' as const,
        content: streamingContent,
        timestamp: Date.now()
      }]);
    }
  });

  const sendMessage = useCallback(async (content: string) => {
    // 添加用户消息
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: Date.now()
    };
    
    setMessages(prev => [...prev, userMessage]);

    // 准备对话历史
    const conversationHistory = messages.map(m => ({
      role: m.role,
      content: m.content
    }));

    // 开始流式请求
    await stream(apiUrl, {
      messages: [...conversationHistory, { role: 'user', content }]
    });
  }, [messages, stream, apiUrl]);

  const clearHistory = useCallback(() => {
    setMessages([]);
  }, []);

  return {
    messages,
    streamingContent,
    isLoading,
    sendMessage,
    clearHistory
  };
}
```

---

## 4.5.4 状态管理 (Zustand)

```tsx
// examples/4-5-frontend/src/stores/chatStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

interface ChatState {
  messages: Message[];
  currentSessionId: string | null;
  sessions: Map<string, Message[]>;
  
  // Actions
  addMessage: (message: Message) => void;
  setMessages: (messages: Message[]) => void;
  createSession: () => string;
  switchSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
  clearAll: () => void;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      messages: [],
      currentSessionId: null,
      sessions: new Map(),
      
      addMessage: (message) => {
        const { currentSessionId, sessions } = get();
        const currentMessages = sessions.get(currentSessionId!) || [];
        
        const updatedMessages = [...currentMessages, message];
        sessions.set(currentSessionId!, updatedMessages);
        
        set({ 
          messages: updatedMessages,
          sessions: new Map(sessions)
        });
      },
      
      setMessages: (messages) => {
        const { currentSessionId, sessions } = get();
        sessions.set(currentSessionId!, messages);
        set({ 
          messages,
          sessions: new Map(sessions)
        });
      },
      
      createSession: () => {
        const sessionId = crypto.randomUUID();
        set({ 
          currentSessionId: sessionId,
          messages: [],
          sessions: new Map(get().sessions.set(sessionId, []))
        });
        return sessionId;
      },
      
      switchSession: (sessionId) => {
        const sessions = get().sessions;
        const messages = sessions.get(sessionId) || [];
        set({ 
          currentSessionId: sessionId,
          messages
        });
      },
      
      deleteSession: (sessionId) => {
        const sessions = new Map(get().sessions);
        sessions.delete(sessionId);
        set({ sessions });
        
        if (get().currentSessionId === sessionId) {
          const firstSession = Array.from(sessions.keys())[0];
          if (firstSession) {
            get().switchSession(firstSession);
          } else {
            get().createSession();
          }
        }
      },
      
      clearAll: () => {
        set({
          messages: [],
          currentSessionId: null,
          sessions: new Map()
        });
      }
    }),
    {
      name: 'chat-storage',
      partialize: (state) => ({
        sessions: Object.fromEntries(state.sessions),
        currentSessionId: state.currentSessionId
      })
    }
  )
);
```

---

## 4.5.5 完整应用示例

```tsx
// examples/4-5-frontend/src/App.tsx
import React from 'react';
import { ChatInput } from './components/ChatInput';
import { MessageList } from './components/MessageList';
import { useChat } from './hooks/useChat';
import { Sidebar } from './components/Sidebar';

const API_URL = '/api/chat';

export default function ChatApp() {
  const {
    messages,
    streamingContent,
    isLoading,
    sendMessage
  } = useChat(API_URL);

  const handleSend = async (content: string) => {
    await sendMessage(content);
  };

  return (
    <div className="chat-app">
      <Sidebar />
      
      <main className="chat-main">
        <header className="chat-header">
          <h1>AI 助手</h1>
        </header>
        
        <div className="chat-container">
          <MessageList 
            messages={messages}
            streamingContent={streamingContent}
            isLoading={isLoading}
          />
          
          <ChatInput 
            onSend={handleSend}
            disabled={isLoading}
          />
        </div>
      </main>
    </div>
  );
}
```

### 样式文件

```css
/* examples/4-5-frontend/src/styles/chat.css */

.chat-app {
  display: flex;
  height: 100vh;
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  max-width: 900px;
  margin: 0 auto;
  width: 100%;
}

.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.message {
  display: flex;
  margin-bottom: 16px;
  animation: fadeIn 0.3s ease;
}

.message.user {
  justify-content: flex-end;
}

.message-bubble {
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 12px;
}

.message.user .message-bubble {
  background: #007bff;
  color: white;
  border-bottom-right-radius: 4px;
}

.message.assistant .message-bubble {
  background: #f1f1f1;
  border-bottom-left-radius: 4px;
}

.chat-input-form {
  display: flex;
  gap: 12px;
  padding: 16px 20px;
  background: white;
  border-top: 1px solid #e0e0e0;
}

.chat-input-textarea {
  flex: 1;
  resize: none;
  padding: 12px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.5;
}

.send-button {
  width: 44px;
  height: 44px;
  border-radius: 8px;
  border: none;
  background: #007bff;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.send-button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

/* 打字动画 */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* 打字指示器 */
.typing-indicator span {
  display: inline-block;
  width: 8px;
  height: 8px;
  background: #999;
  border-radius: 50%;
  margin: 0 2px;
  animation: bounce 1.4s infinite;
}

.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
  0%, 40%, 100% { transform: translateY(0); }
  40% { transform: translateY(-6px); }
}
```

---

## 练习题

1. **流式优化**：如何实现中断正在生成的回复？

2. **性能优化**：大量消息时如何避免渲染卡顿？

3. **离线支持**：如何添加 PWA 和离线缓存？

---

[← 上一节：4.4 Agent 应用开发](4-4-agent.md) | [下一节：4.6 后端 AI 服务 →](4-6-backend-ai.md)
