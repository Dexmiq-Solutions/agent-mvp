import React, { useState } from 'react';
import type { Conversation } from '../types';

interface ConversationListProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  loading: boolean;
  onSelectConversation: (conversationId: string) => void;
  onCreateConversation: (title?: string) => Promise<void>;
}

export const ConversationList: React.FC<ConversationListProps> = ({
  conversations,
  activeConversationId,
  loading,
  onSelectConversation,
  onCreateConversation,
}) => {
  const [showInput, setShowInput] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [creating, setCreating] = useState(false);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    try {
      await onCreateConversation(newTitle.trim() || undefined);
      setNewTitle('');
      setShowInput(false);
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="workspace-panel conversation-panel" id="conversation-panel">
      <div className="panel-header">
        <h3>Conversations</h3>
        {!showInput && (
          <button
            type="button"
            className="btn btn-sm btn-primary"
            onClick={() => setShowInput(true)}
            id="btn-new-conversation"
          >
            + New Conversation
          </button>
        )}
      </div>

      {showInput && (
        <form onSubmit={handleCreate} className="inline-form" id="create-conversation-form">
          <input
            type="text"
            placeholder="Conversation title (optional)"
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            disabled={creating}
            autoFocus
            id="input-conversation-title"
          />
          <div className="inline-form-actions">
            <button
              type="submit"
              className="btn btn-sm btn-primary"
              disabled={creating}
              id="btn-submit-conversation"
            >
              {creating ? 'Creating...' : 'Create'}
            </button>
            <button
              type="button"
              className="btn btn-sm btn-secondary"
              onClick={() => {
                setShowInput(false);
                setNewTitle('');
              }}
              disabled={creating}
            >
              Cancel
            </button>
          </div>
        </form>
      )}

      {loading && conversations.length === 0 ? (
        <div className="panel-loading">Loading conversations...</div>
      ) : conversations.length === 0 ? (
        <div className="panel-empty" id="conversations-empty">
          <p>No conversations yet.</p>
          {!showInput && (
            <button
              type="button"
              className="btn btn-sm btn-outline"
              onClick={() => setShowInput(true)}
            >
              Start First Conversation
            </button>
          )}
        </div>
      ) : (
        <ul className="item-list" id="conversations-list">
          {conversations.map((conv) => {
            const isActive = conv.id === activeConversationId;
            return (
              <li
                key={conv.id}
                className={`list-item ${isActive ? 'item-active' : ''}`}
                onClick={() => onSelectConversation(conv.id)}
                id={`conversation-item-${conv.id}`}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    onSelectConversation(conv.id);
                  }
                }}
              >
                <div className="item-title" title={conv.title}>
                  💬 {conv.title || 'Untitled Conversation'}
                </div>
                <div className="item-meta">
                  {typeof conv.messages_count === 'number' && (
                    <span className="badge badge-meta">
                      {conv.messages_count} msg{conv.messages_count === 1 ? '' : 's'}
                    </span>
                  )}
                  {conv.updated_at && (
                    <span className="timestamp">
                      {new Date(conv.updated_at).toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </span>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};
