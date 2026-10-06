import React, { useState, useEffect, useRef } from 'react';
import {
  type Conversation,
  type Message,
  type WorkflowProgressEvent,
  type AgentActivityItem,
  getMessageWorkflowStatus,
} from '../types';
import { sendMessageStream } from '../api/conversations';
import { AgentActivityView } from './chat/AgentActivityView';

interface ChatViewProps {
  projectId: string;
  conversation: Conversation | null;
  messages: Message[];
  loadingMessages: boolean;
  onRefreshMessages: () => void;
  onMessageSent: (userMsg: Message, assistantMsg: Message) => void;
}

export const ChatView: React.FC<ChatViewProps> = ({
  projectId,
  conversation,
  messages,
  loadingMessages,
  onRefreshMessages,
  onMessageSent,
}) => {
  const [inputText, setInputText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  const [activeProgress, setActiveProgress] = useState<WorkflowProgressEvent | null>(null);
  const [activityLog, setActivityLog] = useState<AgentActivityItem[]>([]);
  const [chatError, setChatError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent, activeProgress]);

  useEffect(() => {
    // Focus input when conversation is selected
    if (conversation && !isGenerating) {
      inputRef.current?.focus();
    }
  }, [conversation?.id, isGenerating]);

  if (!conversation) {
    return (
      <div className="chat-view chat-empty" id="chat-no-selection">
        <div className="chat-empty-content">
          <h3>No Conversation Selected</h3>
          <p>Select a conversation from the list or start a new one to begin chatting with the BRD Agent.</p>
        </div>
      </div>
    );
  }

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const content = inputText.trim();
    if (!content || isGenerating) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setInputText('');
    setChatError(null);
    setIsGenerating(true);
    setStreamingContent('');
    setActiveProgress(null);
    setActivityLog([]);

    // Optimistic user message for immediate UI responsiveness
    const optimisticUserMessage: Message = {
      id: `temp-user-${Date.now()}`,
      conversation_id: conversation.id,
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    };

    let accumulatedContent = '';

    try {
      await sendMessageStream({
        projectId,
        conversationId: conversation.id,
        content,
        signal: controller.signal,
        onToken: (token: string) => {
          accumulatedContent += token;
          setStreamingContent(accumulatedContent);
        },
        onProgress: (prog: WorkflowProgressEvent) => {
          setActiveProgress(prog);
          setActivityLog((prev) => {
            const updated = prev.map((item, idx) => {
              if (idx === prev.length - 1 && item.status === 'in_progress') {
                return { ...item, status: 'completed' as const };
              }
              return item;
            });
            const msgText = prog.message || (prog.section ? `Working on ${prog.section}` : 'Executing agent workflow...');
            if (updated.length > 0 && updated[updated.length - 1].message === msgText) {
              return updated;
            }
            const newItem: AgentActivityItem = {
              id: `act-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
              message: msgText,
              phase: prog.phase,
              section: prog.section,
              actor: prog.actor,
              status: 'in_progress',
              timestamp: new Date().toISOString(),
            };
            return [...updated, newItem];
          });
        },
        onDone: (finalAssistantMessage: Message) => {
          setStreamingContent(null);
          setActiveProgress(null);
          setIsGenerating(false);
          abortControllerRef.current = null;
          onMessageSent(optimisticUserMessage, finalAssistantMessage);
        },
        onError: (err: string) => {
          setChatError(err);
          setIsGenerating(false);
          setStreamingContent(null);
          setActiveProgress(null);
          abortControllerRef.current = null;
          // Refresh messages from server to sync state
          onRefreshMessages();
        },
      });
    } catch (err: unknown) {
      if (controller.signal.aborted) {
        return;
      }
      const errMsg = err instanceof Error ? err.message : 'Failed to send message';
      setChatError(errMsg);
      setIsGenerating(false);
      setStreamingContent(null);
      setActiveProgress(null);
      abortControllerRef.current = null;
      onRefreshMessages();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const lastAssistantMessage = [...messages].reverse().find((m) => m.role === 'assistant');
  const isWaitingForClarification =
    lastAssistantMessage &&
    getMessageWorkflowStatus(lastAssistantMessage) === 'WAITING_FOR_CLARIFICATION';

  return (
    <div className="chat-view" id="chat-view-container">
      <div className="chat-header">
        <div className="chat-title-group">
          <h3>{conversation.title || 'Conversation'}</h3>
          <span className="badge badge-id">ID: {conversation.id.slice(0, 8)}...</span>
        </div>
        <button
          type="button"
          className="btn btn-sm btn-secondary"
          onClick={onRefreshMessages}
          disabled={loadingMessages || isGenerating}
          title="Reload message history"
          id="btn-refresh-chat"
        >
          ↻ History
        </button>
      </div>

      {chatError && (
        <div className="alert alert-error" id="chat-error-banner">
          {chatError}
        </div>
      )}

      <div className="chat-messages" id="chat-messages-container">
        {loadingMessages && messages.length === 0 ? (
          <div className="chat-loading">Loading message history...</div>
        ) : messages.length === 0 && !isGenerating ? (
          <div className="chat-empty-thread" id="chat-empty-thread">
            <p className="lead">Conversation started.</p>
            <p>Ask a question about the project scope, requirements, or uploaded sources.</p>
          </div>
        ) : (
          <>
            {messages.map((msg) => {
              const isUser = msg.role === 'user';
              const workflowStatus = isUser ? 'NORMAL' : getMessageWorkflowStatus(msg);

              return (
                <div
                  key={msg.id}
                  className={`chat-bubble-row ${isUser ? 'row-user' : 'row-agent'}`}
                  id={`msg-${msg.id}`}
                >
                  <div className={`chat-bubble ${isUser ? 'bubble-user' : 'bubble-agent'}`}>
                    <div className="bubble-sender">
                      {isUser ? 'User' : 'BRD Agent'}
                    </div>

                    {!isUser && workflowStatus === 'WAITING_FOR_CLARIFICATION' && (
                      <div className="workflow-status-badge badge-clarification" id={`clarification-badge-${msg.id}`}>
                        <span className="badge-icon">⏳</span>
                        <span><strong>Clarification Required</strong> — Reply below to resume the BRD workflow.</span>
                      </div>
                    )}

                    {!isUser && workflowStatus === 'HALTED' && (
                      <div className="workflow-status-badge badge-halted" id={`halted-badge-${msg.id}`}>
                        <span className="badge-icon">⚠</span>
                        <span><strong>Section Progression Halted</strong> — Review the diagnostic report below.</span>
                      </div>
                    )}

                    {!isUser && workflowStatus === 'COMPLETED' && (
                      <div className="workflow-status-badge badge-completed" id={`completed-badge-${msg.id}`}>
                        <span className="badge-icon">✓</span>
                        <span><strong>BRD Completed</strong></span>
                      </div>
                    )}

                    <div className="bubble-content">{msg.content}</div>
                    {msg.created_at && (
                      <div className="bubble-time">
                        {new Date(msg.created_at).toLocaleTimeString([], {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {isGenerating && (
              <div className="chat-bubble-row row-agent" id="streaming-agent-bubble">
                <div className="chat-bubble bubble-agent" style={{ maxWidth: '85%', width: '100%' }}>
                  <div className="bubble-sender mb-2">
                    BRD Agent <span className="typing-indicator">generating...</span>
                  </div>
                  <AgentActivityView
                    isLive={true}
                    currentProgress={activeProgress}
                    activityLog={activityLog}
                    defaultExpanded={true}
                  />
                  {streamingContent ? (
                    <div className="bubble-content mt-3">{streamingContent}</div>
                  ) : null}
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area" id="chat-input-area">
        <form onSubmit={handleSend} className="chat-input-form">
          <textarea
            ref={inputRef}
            rows={2}
            placeholder={
              isWaitingForClarification && !isGenerating
                ? 'Provide clarification to resume BRD workflow...'
                : isGenerating
                ? 'BRD Agent is generating a response...'
                : 'Type a message... (Press Enter to send, Shift+Enter for new line)'
            }
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isGenerating}
            id="chat-message-input"
          />
          <button
            type="submit"
            className="btn btn-primary btn-send"
            disabled={isGenerating || !inputText.trim()}
            id="btn-send-message"
          >
            {isGenerating ? 'Generating...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
};
