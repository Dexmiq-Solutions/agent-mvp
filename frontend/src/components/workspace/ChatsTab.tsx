import React, { useEffect, useState, useCallback } from 'react';
import { MessageSquare, Trash2, AlertCircle, RotateCcw } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { formatDistanceToNow } from 'date-fns';
import { listConversations, deleteConversation } from '../../services/chatService';
import type { Conversation } from '../../types';

interface ChatsTabProps {
  projectId: string;
}

// ── Conversation card skeleton ────────────────────────────────────────────────
function ConvSkeleton() {
  return (
    <div
      className="skeleton"
      style={{ height: '76px', borderRadius: '10px', flexShrink: 0 }}
    />
  );
}

// ── Delete confirmation dialog ────────────────────────────────────────────────
function DeleteConfirmDialog({
  convTitle,
  onConfirm,
  onCancel,
  deleting,
}: {
  convTitle: string;
  onConfirm: () => void;
  onCancel: () => void;
  deleting: boolean;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !deleting) onCancel(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onCancel, deleting]);

  return (
    <div
      className="backdrop-enter"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 50,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '16px',
        background: 'rgba(0,0,0,0.65)',
        backdropFilter: 'blur(8px)',
      }}
      onClick={(e) => { if (e.target === e.currentTarget && !deleting) onCancel(); }}
    >
      <div
        className="modal-enter"
        style={{
          width: '100%',
          maxWidth: '360px',
          background: '#111111',
          border: '1px solid #222222',
          borderRadius: '12px',
          boxShadow: '0 24px 64px rgba(0,0,0,0.8)',
          padding: '20px',
        }}
      >
        <p style={{ fontSize: '14px', color: '#FFFFFF', fontWeight: 600, marginBottom: '6px' }}>
          Delete conversation?
        </p>
        <p style={{ fontSize: '13px', color: '#8E8E93', lineHeight: 1.55, marginBottom: '20px' }}>
          <span style={{ color: 'rgba(255,255,255,0.7)', fontWeight: 500 }}>"{convTitle}"</span> will be permanently deleted.
        </p>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
          <button
            onClick={onCancel}
            disabled={deleting}
            style={{
              padding: '7px 14px', borderRadius: '7px', border: '1px solid #2a2a2a',
              background: 'transparent', color: '#8E8E93', fontSize: '13px',
              fontFamily: 'Inter, sans-serif', fontWeight: 500, cursor: deleting ? 'not-allowed' : 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={deleting}
            style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              minWidth: '80px', padding: '7px 14px', borderRadius: '7px',
              border: '1px solid rgba(239,68,68,0.3)', background: 'rgba(239,68,68,0.1)',
              color: '#f87171', fontSize: '13px', fontFamily: 'Inter, sans-serif',
              fontWeight: 600, cursor: deleting ? 'not-allowed' : 'pointer',
            }}
          >
            {deleting
              ? <div style={{ width: '14px', height: '14px', border: '2px solid rgba(248,113,113,0.3)', borderTopColor: '#f87171', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
              : 'Delete'}
          </button>
        </div>
      </div>
    </div>
  );
}

export const ChatsTab: React.FC<ChatsTabProps> = ({ projectId }) => {
  const navigate = useNavigate();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<Conversation | null>(null);
  const [deleting, setDeleting] = useState(false);

  const fetchConversations = useCallback(async () => {
    try {
      setError(null);
      setLoading(true);
      const res = await listConversations(projectId, 50, 0);
      setConversations(res.items);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load conversations.');
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => { fetchConversations(); }, [fetchConversations]);

  const handleDelete = async () => {
    if (!pendingDelete) return;
    try {
      setDeleting(true);
      await deleteConversation(projectId, pendingDelete.id);
      setConversations((prev) => prev.filter((c) => c.id !== pendingDelete.id));
      setPendingDelete(null);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to delete conversation.');
    } finally {
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {[0, 1, 2].map((i) => <ConvSkeleton key={i} />)}
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '24px 16px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px', textAlign: 'center' }}>
        <AlertCircle size={20} color="#f87171" strokeWidth={1.75} />
        <p style={{ fontSize: '13px', color: '#8E8E93' }}>Failed to load conversations.</p>
        <button
          onClick={fetchConversations}
          style={{
            display: 'flex', alignItems: 'center', gap: '5px',
            padding: '6px 12px', borderRadius: '7px',
            border: '1px solid #2a2a2a', background: '#141414',
            color: '#FFFFFF', fontSize: '12.5px', fontFamily: 'Inter, sans-serif',
            fontWeight: 500, cursor: 'pointer',
          }}
        >
          <RotateCcw size={12} strokeWidth={2} /> Retry
        </button>
      </div>
    );
  }

  if (conversations.length === 0) {
    return (
      <div style={{ paddingTop: '64px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5 relative
                       bg-gradient-to-b from-zinc-800/50 to-zinc-900/50 backdrop-blur-md border border-white/10
                       shadow-[inset_0_1px_0_rgba(255,255,255,0.1)]"
          >
            <div className="absolute inset-0 bg-indigo-500/20 blur-xl rounded-full" />
            <MessageSquare size={26} strokeWidth={1.5} className="text-zinc-400 relative z-10" />
          </div>
          <p className="text-[15px] font-semibold text-white mb-2">No conversations yet</p>
          <p className="text-[13.5px] text-zinc-400 max-w-[280px] leading-relaxed">
            Type a question above to start chatting with your AI assistant.
          </p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div style={{ paddingTop: '24px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {conversations.map((conv) => (
          <ConversationCard
            key={conv.id}
            conversation={conv}
            projectId={projectId}
            onNavigate={() => navigate(`/projects/${projectId}/c/${conv.id}`)}
            onDeleteRequest={() => setPendingDelete(conv)}
          />
        ))}
      </div>

      {pendingDelete && (
        <DeleteConfirmDialog
          convTitle={pendingDelete.title}
          onConfirm={handleDelete}
          onCancel={() => setPendingDelete(null)}
          deleting={deleting}
        />
      )}
    </>
  );
};

// ── Conversation card ─────────────────────────────────────────────────────────
function ConversationCard({
  conversation,
  onNavigate,
  onDeleteRequest,
}: {
  conversation: Conversation;
  projectId: string;
  onNavigate: () => void;
  onDeleteRequest: () => void;
}) {
  const [hovered, setHovered] = useState(false);

  const updatedAt = conversation.updated_at
    ? formatDistanceToNow(new Date(conversation.updated_at), { addSuffix: true })
    : null;

  return (
    <div
      onClick={onNavigate}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      role="button"
      tabIndex={0}
      aria-label={`Open conversation: ${conversation.title}`}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onNavigate(); } }}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '18px 22px',
        borderRadius: '14px',
        border: hovered ? '1px solid rgba(255,255,255,0.15)' : '1px solid #1e1e1e',
        background: hovered ? '#171717' : '#111111',
        cursor: 'pointer',
        transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
        gap: '12px',
        boxShadow: hovered ? '0 4px 12px rgba(0,0,0,0.2)' : 'none',
        transform: hovered ? 'translateY(-1px)' : 'translateY(0)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', minWidth: 0 }}>
        <div
          style={{
            width: '40px', height: '40px', borderRadius: '12px', flexShrink: 0,
            background: hovered ? 'rgba(99,102,241,0.08)' : '#171717',
            border: hovered ? '1px solid rgba(99,102,241,0.2)' : '1px solid #222',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'all 0.15s ease',
          }}
        >
          <MessageSquare
            size={18}
            strokeWidth={1.75}
            color={hovered ? '#a5b4fc' : '#8E8E93'}
            style={{ transition: 'color 0.15s ease' }}
          />
        </div>
        <div style={{ minWidth: 0 }}>
          <p
            style={{
              fontSize: '14.5px', fontWeight: 500, color: '#FFFFFF',
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              marginBottom: '4px', letterSpacing: '-0.01em',
            }}
          >
            {conversation.title}
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            {updatedAt && (
              <span style={{ fontSize: '12px', color: '#8E8E93' }}>Updated {updatedAt}</span>
            )}
            {updatedAt && conversation.messages_count > 0 && (
              <span style={{ fontSize: '12px', color: '#444' }}>·</span>
            )}
            {conversation.messages_count > 0 && (
              <span style={{ fontSize: '12px', color: '#8E8E93' }}>
                {conversation.messages_count} {conversation.messages_count === 1 ? 'message' : 'messages'}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Delete button — visible on hover */}
      <button
        onClick={(e) => { e.stopPropagation(); onDeleteRequest(); }}
        aria-label="Delete conversation"
        title="Delete conversation"
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          width: '32px', height: '32px', borderRadius: '8px',
          border: '1px solid transparent', background: 'transparent',
          color: hovered ? 'rgba(252,165,165,0.7)' : 'transparent',
          cursor: 'pointer', flexShrink: 0,
          transition: 'all 0.15s ease',
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLButtonElement).style.color = '#f87171';
          (e.currentTarget as HTMLButtonElement).style.background = 'rgba(239,68,68,0.1)';
          (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(239,68,68,0.2)';
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLButtonElement).style.color = hovered ? 'rgba(252,165,165,0.7)' : 'transparent';
          (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
          (e.currentTarget as HTMLButtonElement).style.borderColor = 'transparent';
        }}
      >
        <Trash2 size={14} strokeWidth={2} />
      </button>
    </div>
  );
}
