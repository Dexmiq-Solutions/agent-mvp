// =============================================================================
// ConversationView — Step 6: Chat Interaction States
// =============================================================================

import React, {
  useEffect,
  useState,
  useRef,
  useCallback,
  useMemo,
} from 'react';
import { useParams, useLocation, Link, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  MessageSquare,
  Send,
  Check,
  X,
  AlertCircle,
  RotateCcw,
  Trash2,
  Edit2,
  Bot,
  Copy,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Components } from 'react-markdown';
import { getConversation, sendMessage, updateConversation, deleteConversation } from '../services/chatService';
import type { Conversation, ChatStatus, UIMessage } from '../types';
import { TelemetryView } from '../components/chat/TelemetryView';

// ─── Helpers ─────────────────────────────────────────────────────────────────

type LoadState = 'loading' | 'not_found' | 'error' | 'loaded';

/** Classify an API error into a friendly message string. */
function classifyError(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err);
  if (msg.includes('422')) {
    return 'I could not find sufficient grounded information in your uploaded project sources to answer this question.';
  }
  if (msg.includes('429')) {
    return 'AI service is currently busy. Please retry in a few seconds.';
  }
  if (msg.includes('504') || msg.toLowerCase().includes('timeout')) {
    return 'Request timed out. Please try again.';
  }
  if (msg.includes('404')) {
    return '404_NOT_FOUND'; // special sentinel
  }
  if (msg.toLowerCase().includes('network') || msg.toLowerCase().includes('failed to fetch')) {
    return 'Network error. Check your connection and retry.';
  }
  return 'Something went wrong while generating the response.';
}

/** Check whether the user is scrolled near the bottom of a container. */
function isNearBottom(el: HTMLElement, threshold = 120): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function PageErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-12 text-center">
      <div className="flex items-start gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/20 max-w-sm text-left">
        <AlertCircle size={16} className="text-red-400 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <p className="text-[13px] font-medium text-red-400 mb-1">Error</p>
          <p className="text-[12.5px] text-red-300/70">{message}</p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-red-500/20 bg-red-500/10 text-red-300/80 text-xs font-medium hover:bg-red-500/20 transition-colors"
          >
            <RotateCcw size={11} /> Retry
          </button>
        )}
      </div>
    </div>
  );
}

function NotFound() {
  const navigate = useNavigate();
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-12 text-center">
      <div className="w-12 h-12 rounded-xl bg-[#141414] border border-[#222] flex items-center justify-center mb-4">
        <MessageSquare size={20} className="text-[#8E8E93]" />
      </div>
      <h1 className="text-[17px] font-bold text-white mb-1.5 tracking-tight">Workspace Not Found</h1>
      <p className="text-[13px] text-[#8E8E93] max-w-[260px] leading-relaxed mb-5">
        The requested conversation could not be found.
      </p>
      <button
        onClick={() => navigate(-1)}
        className="px-4 py-1.5 rounded-lg border border-indigo-500/35 bg-indigo-500/10 text-indigo-300 text-[13px] font-medium hover:bg-indigo-500/20 transition-colors"
      >
        Return to Projects
      </button>
    </div>
  );
}

function ConversationSkeleton() {
  return (
    <div className="flex-1 flex flex-col bg-[#0A0A0A] overflow-hidden">
      <div className="h-[52px] flex items-center px-5 border-b border-[#1e1e1e] flex-shrink-0 gap-2.5">
        <div className="skeleton w-7 h-7 rounded-md" />
        <div className="skeleton w-6 h-6 rounded-md" />
        <div className="skeleton w-32 h-3.5 rounded" />
      </div>
      <div className="flex-1 p-6 space-y-6">
        <div className="flex flex-col gap-2 items-end">
          <div className="skeleton w-1/3 h-12 rounded-2xl rounded-tr-sm" />
        </div>
        <div className="flex flex-col gap-2 items-start">
          <div className="skeleton w-1/2 h-24 rounded-2xl rounded-tl-sm" />
        </div>
      </div>
    </div>
  );
}

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
      className="backdrop-enter fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/65 backdrop-blur-[8px]"
      onClick={(e) => { if (e.target === e.currentTarget && !deleting) onCancel(); }}
    >
      <div className="modal-enter w-full max-w-[360px] bg-[#111] border border-[#222] rounded-xl shadow-[0_24px_64px_rgba(0,0,0,0.8)] p-5">
        <p className="text-[14px] text-white font-semibold mb-1.5">Delete conversation?</p>
        <p className="text-[13px] text-[#8E8E93] leading-[1.55] mb-5">
          <span className="text-white/70 font-medium">"{convTitle}"</span> will be permanently deleted. This action cannot be undone.
        </p>
        <div className="flex justify-end gap-2">
          <button
            onClick={onCancel}
            disabled={deleting}
            className="px-3.5 py-1.5 rounded-lg border border-[#2a2a2a] text-[#8E8E93] text-[13px] font-medium hover:bg-[#1a1a1a] transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={deleting}
            className="flex items-center justify-center min-w-[80px] px-3.5 py-1.5 rounded-lg border border-red-500/30 bg-red-500/10 text-red-400 text-[13px] font-semibold hover:bg-red-500/20 transition-colors disabled:opacity-50"
          >
            {deleting ? (
              <div className="w-3.5 h-3.5 border-2 border-red-400/30 border-t-red-400 rounded-full animate-spin" />
            ) : (
              'Delete'
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Code block with copy button ─────────────────────────────────────────────

function CodeBlock({ children, className }: { children?: React.ReactNode; className?: string }) {
  const [copied, setCopied] = useState(false);
  const lang = className?.replace('language-', '') ?? '';

  const handleCopy = () => {
    const text = typeof children === 'string' ? children : String(children ?? '');
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  };

  return (
    <div className="relative group my-3 rounded-xl border border-[#2a2a2a] bg-[#141414] overflow-hidden">
      {/* Language tag + copy button */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-[#222] bg-[#111]">
        <span className="text-[10.5px] text-[#555] font-mono">{lang || 'code'}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 px-2 py-0.5 rounded-md text-[10.5px] font-medium text-[#666] hover:text-[#aaa] hover:bg-white/5 transition-all"
          aria-label="Copy code"
        >
          {copied ? (
            <>
              <Check size={10} className="text-emerald-400" />
              <span className="text-emerald-400">Copied</span>
            </>
          ) : (
            <>
              <Copy size={10} />
              Copy
            </>
          )}
        </button>
      </div>
      <pre className="overflow-x-auto px-4 py-3 text-[13px] leading-relaxed text-[#d4d4d4] font-mono whitespace-pre">
        <code>{children}</code>
      </pre>
    </div>
  );
}

/** react-markdown component overrides */
const markdownComponents: Components = {
  // Fenced code blocks
  code({ className, children, ...props }) {
    const isBlock = className?.startsWith('language-');
    if (isBlock) {
      return <CodeBlock className={className}>{children}</CodeBlock>;
    }
    return (
      <code
        className="px-1.5 py-0.5 rounded-md bg-[#1e1e1e] border border-[#2a2a2a] text-[12.5px] font-mono text-[#e2c08d]"
        {...props}
      >
        {children}
      </code>
    );
  },
  // Pre: already handled inside CodeBlock, so just pass-through
  pre({ children }) {
    return <>{children}</>;
  },
  // Links open in new tab
  a({ href, children }) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-indigo-400 underline underline-offset-2 hover:text-indigo-300 transition-colors"
      >
        {children}
      </a>
    );
  },
};

// ─── ThinkingIndicator ────────────────────────────────────────────────────────

function ThinkingIndicator() {
  return (
    <div className="flex w-full justify-start slide-up">
      <div
        className="w-8 h-8 rounded-lg flex items-center justify-center mr-3 mt-0.5 flex-shrink-0"
        style={{ background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.2)' }}
      >
        <Bot size={15} style={{ color: '#818cf8' }} />
      </div>
      <div
        className="flex items-center gap-1.5 px-4"
        style={{
          height: '40px',
          borderRadius: '16px',
          borderTopLeftRadius: '4px',
          background: '#141414',
          border: '1px solid #242424',
        }}
      >
        <span className="dot-pulse-dot w-1.5 h-1.5 rounded-full" style={{ background: 'rgba(129,140,248,0.7)', display: 'inline-block' }} />
        <span className="dot-pulse-dot w-1.5 h-1.5 rounded-full" style={{ background: 'rgba(129,140,248,0.7)', display: 'inline-block' }} />
        <span className="dot-pulse-dot w-1.5 h-1.5 rounded-full" style={{ background: 'rgba(129,140,248,0.7)', display: 'inline-block' }} />
        <span className="text-[12px] ml-1.5" style={{ color: 'rgba(129,140,248,0.6)' }}>Thinking…</span>
      </div>
    </div>
  );
}

// ─── MessageTurn ──────────────────────────────────────────────────────────────

interface MessageTurnProps {
  msg: UIMessage;
  /** If true, generation is active — disables retry button to prevent double-fire */
  generationActive: boolean;
  onRetry: (originalText: string) => void;
}

function MessageTurn({ msg, generationActive, onRetry }: MessageTurnProps) {
  const isUser = msg.role === 'user';

  if (isUser) {
    return (
      <div className="flex w-full justify-end slide-up">
        <div style={{ maxWidth: 'min(85%, 720px)', marginLeft: '3rem' }}>
          <div
            style={{
              background: '#1c1c1e',
              border: '1px solid #2c2c2e',
              borderRadius: '18px',
              borderTopRightRadius: '4px',
              padding: '10px 16px',
              fontSize: '14px',
              lineHeight: '1.65',
              color: '#f5f5f7',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {msg.content}
          </div>
        </div>
      </div>
    );
  }

  // Assistant error turn
  if (msg._error) {
    return (
      <div className="flex w-full justify-start slide-up">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center mr-3 mt-0.5 flex-shrink-0"
          style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.18)' }}
        >
          <AlertCircle size={15} style={{ color: '#f87171' }} />
        </div>
        <div style={{ maxWidth: 'min(85%, 720px)', marginRight: '1rem' }}>
          <div
            style={{
              background: 'rgba(239,68,68,0.06)',
              border: '1px solid rgba(239,68,68,0.18)',
              borderRadius: '16px',
              borderTopLeftRadius: '4px',
              padding: '12px 16px',
            }}
          >
            <p style={{ fontSize: '13.5px', color: 'rgba(252,165,165,0.9)', lineHeight: '1.6' }}>{msg._errorText}</p>
          </div>
          <div style={{ marginTop: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              disabled={generationActive}
              onClick={() => {
                onRetry(msg.content);
              }}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px',
                padding: '5px 10px',
                borderRadius: '7px',
                border: '1px solid #2a2a2a',
                background: 'transparent',
                color: '#8E8E93',
                fontSize: '12px',
                fontFamily: 'Inter, sans-serif',
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'all 0.12s ease',
                opacity: generationActive ? 0.4 : 1,
              }}
              onMouseEnter={(e) => {
                if (!generationActive) {
                  (e.currentTarget as HTMLButtonElement).style.background = '#1a1a1a';
                  (e.currentTarget as HTMLButtonElement).style.color = '#fff';
                  (e.currentTarget as HTMLButtonElement).style.borderColor = '#333';
                }
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
                (e.currentTarget as HTMLButtonElement).style.color = '#8E8E93';
                (e.currentTarget as HTMLButtonElement).style.borderColor = '#2a2a2a';
              }}
              aria-label="Retry generation"
            >
              <RotateCcw size={11} />
              Retry generation
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Normal assistant turn
  return (
    <div className="flex w-full justify-start slide-up">
      <div
        className="w-8 h-8 rounded-lg flex items-center justify-center mr-3 mt-0.5 flex-shrink-0"
        style={{ background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.2)' }}
      >
        <Bot size={15} style={{ color: '#818cf8' }} />
      </div>
      <div style={{ maxWidth: 'min(85%, 760px)', marginRight: '0.5rem', minWidth: 0 }}>
        <div
          className="chat-prose"
          style={{ fontSize: '14px', lineHeight: '1.7', color: '#d4d4d8' }}
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
            {msg.content}
          </ReactMarkdown>
        </div>

        {/* Telemetry — only for non-error, non-optimistic assistant messages */}
        {!msg._optimistic && msg.metadata && Object.keys(msg.metadata).length > 0 && (
          <TelemetryView metadata={msg.metadata} />
        )}
      </div>
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export const ConversationView: React.FC = () => {
  const { projectId, conversationId } = useParams<{ projectId: string; conversationId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const initialPrompt = (location.state as { initialPrompt?: string } | null)?.initialPrompt;

  // ── Page-level load state
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [pageErrorMsg, setPageErrorMsg] = useState<string | null>(null);
  const [conversation, setConversation] = useState<Conversation | null>(null);

  // ── Messages
  const [messages, setMessages] = useState<UIMessage[]>([]);

  // ── Chat state machine
  const [chatStatus, setChatStatus] = useState<ChatStatus>('idle');
  /**
   * retryText — original user text of the last failed turn.
   * Stored so retry doesn't duplicate the user bubble.
   */
  const retryTextRef = useRef<string | null>(null);

  // ── Composer
  const [inputMessage, setInputMessage] = useState('');

  // ── Header actions
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleInput, setTitleInput] = useState('');
  const [savingTitle, setSavingTitle] = useState(false);
  const [pendingDelete, setPendingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // ── Scroll refs
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  /** Whether generation is in-flight (sending or thinking) */
  const generationActive = chatStatus === 'sending' || chatStatus === 'thinking';

  // ── Scroll helpers

  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'smooth') => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  }, []);

  /** Scroll to bottom only when user hasn't scrolled up intentionally */
  const scrollToBottomIfNear = useCallback(() => {
    const container = scrollContainerRef.current;
    if (container && isNearBottom(container)) {
      scrollToBottom('smooth');
    }
  }, [scrollToBottom]);

  // ── Fetch conversation

  const fetchConversation = useCallback(async () => {
    if (!projectId || !conversationId) return;
    try {
      setLoadState('loading');
      setPageErrorMsg(null);
      const data = await getConversation(projectId, conversationId);
      setConversation(data);
      setMessages((data.messages || []) as UIMessage[]);
      setTitleInput(data.title);
      setLoadState('loaded');
      // Scroll to bottom after history loads (instant)
      requestAnimationFrame(() => scrollToBottom('instant' as ScrollBehavior));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load conversation.';
      if (msg.includes('404') || msg.toLowerCase().includes('not found')) {
        setLoadState('not_found');
      } else {
        setPageErrorMsg(msg);
        setLoadState('error');
      }
    }
  }, [projectId, conversationId, scrollToBottom]);

  useEffect(() => { fetchConversation(); }, [fetchConversation]);

  // ── Auto-send initial prompt when arriving from ProjectHome
  const initialSentRef = useRef(false);
  useEffect(() => {
    if (
      loadState === 'loaded' &&
      initialPrompt &&
      messages.length === 0 &&
      !initialSentRef.current
    ) {
      initialSentRef.current = true;
      navigate('.', { replace: true, state: {} });
      // Small delay to let the view settle
      setTimeout(() => handleSendMessage(initialPrompt), 120);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadState]);

  // ── Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [inputMessage]);

  // ── Scroll when messages or status change
  useEffect(() => {
    scrollToBottomIfNear();
  }, [messages, chatStatus, scrollToBottomIfNear]);

  // ── Core: send a message and generate AI response

  const handleSendMessage = useCallback(async (text: string) => {
    if (!projectId || !conversationId) return;
    const trimmed = text.trim();
    if (!trimmed || generationActive) return;

    // 1. Clear input immediately
    setInputMessage('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    // 2. Optimistic user message
    const tempId = `opt-${crypto.randomUUID()}`;
    const optimisticUserMsg: UIMessage = {
      id: tempId,
      conversation_id: conversationId,
      role: 'user',
      content: trimmed,
      created_at: new Date().toISOString(),
      metadata: {},
      _optimistic: true,
    };

    setMessages((prev) => [...prev, optimisticUserMsg]);
    setChatStatus('sending');

    // Brief "sending" phase before the request fires
    await new Promise((r) => setTimeout(r, 80));
    setChatStatus('thinking');

    // Store in case we need to retry
    retryTextRef.current = trimmed;

    try {
      const assistantMsg = await sendMessage(
        projectId,
        conversationId,
        { content: trimmed },
        true,
      );

      setMessages((prev) => {
        // Confirm optimistic user message (keep it, just remove flag)
        const confirmed = prev.map((m) =>
          m.id === tempId ? { ...m, _optimistic: false } : m
        );
        // Append real assistant response
        return [...confirmed, assistantMsg as UIMessage];
      });

      setChatStatus('idle');
      retryTextRef.current = null;
    } catch (err: unknown) {
      const errText = classifyError(err);

      if (errText === '404_NOT_FOUND') {
        setLoadState('not_found');
        return;
      }

      // Create an error assistant bubble — content = original user text for retry
      const errorMsg: UIMessage = {
        id: `err-${crypto.randomUUID()}`,
        conversation_id: conversationId,
        role: 'assistant',
        content: trimmed, // kept for retry
        created_at: new Date().toISOString(),
        metadata: { error: true },
        _error: true,
        _errorText: errText,
      };

      setMessages((prev) => [
        // Keep optimistic user message confirmed
        ...prev.map((m) => (m.id === tempId ? { ...m, _optimistic: false } : m)),
        errorMsg,
      ]);

      setChatStatus('error');
    }
  }, [projectId, conversationId, generationActive]);

  // ── Retry: re-run generation without adding a new user bubble

  const handleRetry = useCallback(async (originalText: string) => {
    if (!projectId || !conversationId) return;
    if (generationActive) return;

    // Remove the last error assistant message
    setMessages((prev) => {
      const lastIdx = [...prev].reverse().findIndex((m) => m._error);
      if (lastIdx === -1) return prev;
      const realIdx = prev.length - 1 - lastIdx;
      return prev.filter((_, i) => i !== realIdx);
    });

    setChatStatus('thinking');
    retryTextRef.current = originalText;

    try {
      const assistantMsg = await sendMessage(
        projectId,
        conversationId,
        { content: originalText },
        true,
      );

      setMessages((prev) => [...prev, assistantMsg as UIMessage]);
      setChatStatus('idle');
      retryTextRef.current = null;
    } catch (err: unknown) {
      const errText = classifyError(err);

      if (errText === '404_NOT_FOUND') {
        setLoadState('not_found');
        return;
      }

      const errorMsg: UIMessage = {
        id: `err-${crypto.randomUUID()}`,
        conversation_id: conversationId,
        role: 'assistant',
        content: originalText,
        created_at: new Date().toISOString(),
        metadata: { error: true },
        _error: true,
        _errorText: errText,
      };

      setMessages((prev) => [...prev, errorMsg]);
      setChatStatus('error');
    }
  }, [projectId, conversationId, generationActive]);

  // ── Keyboard handler

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(inputMessage);
    }
  };

  // ── Title editing

  const saveTitle = async () => {
    if (!projectId || !conversationId || !conversation) return;
    const t = titleInput.trim();
    if (!t || t === conversation.title) {
      setEditingTitle(false);
      setTitleInput(conversation.title);
      return;
    }
    try {
      setSavingTitle(true);
      const updated = await updateConversation(projectId, conversationId, { title: t });
      setConversation(updated);
      setEditingTitle(false);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to rename conversation');
      setTitleInput(conversation.title);
    } finally {
      setSavingTitle(false);
    }
  };

  // ── Delete conversation

  const handleDelete = async () => {
    if (!projectId || !conversationId || !conversation) return;
    try {
      setDeleting(true);
      await deleteConversation(projectId, conversationId);
      navigate(`/projects/${projectId}`);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to delete conversation');
      setDeleting(false);
      setPendingDelete(false);
    }
  };

  // ── Derived UI values

  const canSend = useMemo(
    () => inputMessage.trim().length > 0 && !generationActive,
    [inputMessage, generationActive],
  );

  const composerPlaceholder = generationActive
    ? 'Waiting for AI to respond…'
    : 'Message AI assistant…';

  // ── Render guards

  if (loadState === 'not_found') {
    return <div className="flex flex-col flex-1 h-full bg-[#0A0A0A]"><NotFound /></div>;
  }
  if (loadState === 'error') {
    return (
      <div className="flex flex-col flex-1 h-full bg-[#0A0A0A]">
        <PageErrorState message={pageErrorMsg || ''} onRetry={fetchConversation} />
      </div>
    );
  }
  if (loadState === 'loading' || !conversation) {
    return <ConversationSkeleton />;
  }

  // ── Main render

  return (
    <div className="flex flex-col flex-1 h-full bg-[#0A0A0A] overflow-hidden text-white">

      {/* ─ Header ─────────────────────────────────────────────────────────── */}
      <header className="flex items-center gap-2.5 h-[52px] px-5 border-b border-[#1e1e1e] flex-shrink-0 bg-[#0A0A0A] z-10">
        <Link
          to={`/projects/${projectId}`}
          className="flex items-center justify-center w-7 h-7 rounded-md border border-[#222] bg-[#141414] text-[#8E8E93] hover:text-white hover:border-[#333] transition-all"
          aria-label="Back to workspace"
        >
          <ArrowLeft size={13} strokeWidth={2} />
        </Link>
        <div className="w-[26px] h-[26px] rounded-md bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center flex-shrink-0">
          <MessageSquare size={13} strokeWidth={1.75} className="text-indigo-300" />
        </div>

        {editingTitle ? (
          <div className="flex items-center gap-1.5 flex-1 min-w-0">
            <input
              // eslint-disable-next-line jsx-a11y/no-autofocus
              autoFocus
              className="bg-[#141414] border border-[#333] text-[13.5px] font-medium text-white px-2 py-1 rounded outline-none focus:border-indigo-500/50 flex-1 max-w-[300px]"
              value={titleInput}
              onChange={(e) => setTitleInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') saveTitle();
                if (e.key === 'Escape') { setEditingTitle(false); setTitleInput(conversation.title); }
              }}
              disabled={savingTitle}
            />
            <button onClick={saveTitle} disabled={savingTitle} className="text-emerald-400 hover:text-emerald-300 p-1">
              <Check size={14} strokeWidth={2} />
            </button>
            <button
              onClick={() => { setEditingTitle(false); setTitleInput(conversation.title); }}
              disabled={savingTitle}
              className="text-red-400 hover:text-red-300 p-1"
            >
              <X size={14} strokeWidth={2} />
            </button>
          </div>
        ) : (
          <div
            className="flex items-center gap-2 flex-1 min-w-0 group cursor-pointer"
            onClick={() => setEditingTitle(true)}
          >
            <h1 className="text-[13.5px] font-medium text-white tracking-tight truncate">
              {conversation.title}
            </h1>
            <Edit2 size={12} className="text-[#8E8E93] opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
        )}

        <button
          onClick={() => setPendingDelete(true)}
          className="w-7 h-7 flex items-center justify-center rounded-md text-[#8E8E93] hover:text-red-400 hover:bg-red-500/10 hover:border-red-500/20 border border-transparent transition-all ml-auto"
          title="Delete Conversation"
        >
          <Trash2 size={14} strokeWidth={2} />
        </button>
      </header>

      {/* ─ Messages Area ───────────────────────────────────────────────────── */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto scroll-smooth"
      >
        <div className="max-w-[1000px] mx-auto px-4 py-8 flex flex-col gap-8">

          {/* Empty state */}
          {messages.length === 0 && !generationActive && (
            <div className="flex flex-col items-center justify-center py-16 text-center slide-up">
              <div
                style={{
                  width: '52px',
                  height: '52px',
                  borderRadius: '16px',
                  background: 'rgba(99,102,241,0.08)',
                  border: '1px solid rgba(99,102,241,0.18)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '20px',
                }}
              >
                <Bot size={22} style={{ color: '#818cf8' }} />
              </div>
              <p style={{ fontSize: '15px', fontWeight: 600, color: '#ffffff', marginBottom: '8px', letterSpacing: '-0.02em' }}>
                Start a conversation
              </p>
              <p style={{ fontSize: '13px', color: '#8E8E93', maxWidth: '280px', lineHeight: 1.65 }}>
                Ask anything about your project. The AI will search your uploaded sources to answer.
              </p>
            </div>
          )}

          {/* Message list */}
          {messages.map((msg) => (
            <MessageTurn
              key={msg.id}
              msg={msg}
              generationActive={generationActive}
              onRetry={handleRetry}
            />
          ))}

          {/* AI thinking indicator */}
          {generationActive && <ThinkingIndicator />}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* ─ Composer ────────────────────────────────────────────────────────── */}
      <div
        style={{
          padding: '16px 20px 20px',
          background: '#0A0A0A',
          borderTop: '1px solid #1e1e1e',
          flexShrink: 0,
        }}
      >
        <div style={{ maxWidth: '860px', margin: '0 auto' }}>
          {/* Composer box */}
          <div
            style={{
              position: 'relative',
              background: '#141414',
              border: generationActive ? '1px solid #1e1e1e' : '1px solid #2a2a2a',
              borderRadius: '20px',
              transition: 'border-color 0.15s ease, box-shadow 0.15s ease',
              opacity: generationActive ? 0.6 : 1,
            }}
            onFocusCapture={(e) => {
              if (!generationActive) {
                (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(99,102,241,0.5)';
                (e.currentTarget as HTMLDivElement).style.boxShadow = '0 0 0 3px rgba(99,102,241,0.08)';
              }
            }}
            onBlurCapture={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget as Node)) {
                (e.currentTarget as HTMLDivElement).style.borderColor = generationActive ? '#1e1e1e' : '#2a2a2a';
                (e.currentTarget as HTMLDivElement).style.boxShadow = 'none';
              }
            }}
          >
            <textarea
              ref={textareaRef}
              id="chat-composer"
              style={{
                display: 'block',
                width: '100%',
                background: 'transparent',
                border: 'none',
                outline: 'none',
                resize: 'none',
                padding: '13px 52px 13px 16px',
                fontSize: '14px',
                lineHeight: '1.6',
                color: '#ffffff',
                fontFamily: 'Inter, sans-serif',
                minHeight: '50px',
                maxHeight: '200px',
                overflow: 'auto',
              }}
              placeholder={composerPlaceholder}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={generationActive}
              rows={1}
              aria-label="Message input"
            />
            <button
              id="chat-send-btn"
              onClick={() => handleSendMessage(inputMessage)}
              disabled={!canSend}
              style={{
                position: 'absolute',
                right: '10px',
                bottom: '9px',
                width: '32px',
                height: '32px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderRadius: '9px',
                border: 'none',
                background: canSend ? '#6366f1' : '#1e1e1e',
                color: canSend ? '#ffffff' : '#444',
                cursor: canSend ? 'pointer' : 'not-allowed',
                transition: 'background 0.15s ease, transform 0.1s ease',
                flexShrink: 0,
              }}
              onMouseEnter={(e) => {
                if (canSend) {
                  (e.currentTarget as HTMLButtonElement).style.background = '#818cf8';
                  (e.currentTarget as HTMLButtonElement).style.transform = 'scale(1.05)';
                }
              }}
              onMouseLeave={(e) => {
                if (canSend) {
                  (e.currentTarget as HTMLButtonElement).style.background = '#6366f1';
                  (e.currentTarget as HTMLButtonElement).style.transform = 'scale(1)';
                }
              }}
              aria-label="Send message"
            >
              {chatStatus === 'sending' ? (
                <div style={{ width: '14px', height: '14px', border: '2px solid rgba(255,255,255,0.25)', borderTopColor: '#fff', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
              ) : (
                <Send size={14} strokeWidth={2.5} style={{ marginLeft: '1px' }} />
              )}
            </button>
          </div>

          {/* Hint text */}
          <p style={{ marginTop: '7px', textAlign: 'center', fontSize: '11px', color: '#383838' }}>
            <span style={{ color: '#4a4a4a', fontWeight: 500 }}>Enter</span> to send
            {'  ·  '}
            <span style={{ color: '#4a4a4a', fontWeight: 500 }}>Shift + Enter</span> for new line
          </p>
        </div>
      </div>

      {/* ─ Delete dialog ───────────────────────────────────────────────────── */}
      {pendingDelete && (
        <DeleteConfirmDialog
          convTitle={conversation.title}
          onConfirm={handleDelete}
          onCancel={() => setPendingDelete(false)}
          deleting={deleting}
        />
      )}
    </div>
  );
};
