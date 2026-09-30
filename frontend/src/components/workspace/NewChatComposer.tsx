import React, { useState, useRef, useCallback } from 'react';
import { ArrowUp, Brain, Mic, AudioWaveform, X } from 'lucide-react';

interface NewChatComposerProps {
  projectName: string;
  onSubmit: (prompt: string) => Promise<void>;
}

// ── Voice/audio coming soon notice ───────────────────────────────────────────
function VoiceNotice({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div
      className="slide-up"
      style={{
        position: 'absolute',
        bottom: 'calc(100% + 8px)',
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 50,
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '8px 12px',
        background: '#1a1a1a',
        border: '1px solid #2e2e2e',
        borderRadius: '8px',
        boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        fontSize: '12.5px',
        color: '#8E8E93',
        whiteSpace: 'nowrap',
      }}
    >
      <span>Voice inputs coming soon</span>
      <button
        onClick={onDismiss}
        aria-label="Dismiss"
        style={{ background: 'none', border: 'none', color: '#8E8E93', cursor: 'pointer', display: 'flex', padding: 0 }}
      >
        <X size={12} />
      </button>
    </div>
  );
}

export const NewChatComposer: React.FC<NewChatComposerProps> = ({ projectName, onSubmit }) => {
  const [input, setInput] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [showVoiceNotice, setShowVoiceNotice] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  const adjustHeight = useCallback(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    adjustHeight();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = async () => {
    const trimmed = input.trim();
    if (!trimmed || submitting) return;
    try {
      setSubmitting(true);
      await onSubmit(trimmed);
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    } finally {
      setSubmitting(false);
    }
  };

  const canSubmit = input.trim().length > 0 && !submitting;

  return (
    <div
      style={{
        background: '#141414',
        border: '1px solid #222222',
        borderRadius: '20px',
        transition: 'border-color 0.15s ease, box-shadow 0.15s ease',
        boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
        display: 'flex',
        flexDirection: 'column',
      }}
      onFocusCapture={(e) => {
        const el = e.currentTarget as HTMLDivElement;
        el.style.borderColor = 'rgba(99,102,241,0.5)';
        el.style.boxShadow = '0 4px 12px rgba(0,0,0,0.4), 0 0 0 1px rgba(99,102,241,0.2)';
      }}
      onBlurCapture={(e) => {
        // Only reset if focus leaves the composer entirely
        if (!e.currentTarget.contains(e.relatedTarget as Node)) {
          const el = e.currentTarget as HTMLDivElement;
          el.style.borderColor = '#222222';
          el.style.boxShadow = '0 4px 12px rgba(0,0,0,0.2)';
        }
      }}
    >
      {/* Textarea */}
      <div style={{ padding: '16px 20px 12px', flex: 1 }}>
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={`+ New chat in ${projectName}...`}
          disabled={submitting}
          rows={2}
          aria-label="New chat message"
          style={{
            width: '100%',
            background: 'transparent',
            border: 'none',
            outline: 'none',
            resize: 'none',
            color: '#FFFFFF',
            fontSize: '15px',
            fontFamily: 'Inter, sans-serif',
            lineHeight: '1.6',
            minHeight: '48px',
            maxHeight: '200px',
            overflow: 'auto',
          }}
        />
      </div>

      {/* Control bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '10px 16px 16px',
        }}
      >
        {/* Left controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', position: 'relative' }}>
          {/* Think button */}
          <button
            onClick={() => {}}
            style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              padding: '6px 12px', borderRadius: '18px',
              border: '1px solid #2a2a2a', background: '#1a1a1a',
              color: '#a5b4fc', fontSize: '13px', fontWeight: 500,
              cursor: 'pointer', transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLButtonElement).style.background = '#222222';
              (e.currentTarget as HTMLButtonElement).style.borderColor = '#333333';
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLButtonElement).style.background = '#1a1a1a';
              (e.currentTarget as HTMLButtonElement).style.borderColor = '#2a2a2a';
            }}
          >
            <Brain size={14} strokeWidth={2} />
            Think
          </button>

          <div style={{ width: '1px', height: '16px', background: '#2a2a2a', margin: '0 4px' }} />

          <div style={{ position: 'relative' }}>
            <ControlButton
              icon={<Mic size={15} strokeWidth={2} />}
              label="Voice input (coming soon)"
              onClick={() => setShowVoiceNotice((v) => !v)}
            />
            {showVoiceNotice && (
              <>
                <div
                  style={{ position: 'fixed', inset: 0, zIndex: 49 }}
                  onClick={() => setShowVoiceNotice(false)}
                />
                <VoiceNotice onDismiss={() => setShowVoiceNotice(false)} />
              </>
            )}
          </div>
          <ControlButton
            icon={<AudioWaveform size={15} strokeWidth={2} />}
            label="Audio waveform (coming soon)"
            onClick={() => setShowVoiceNotice(true)}
          />
        </div>

        {/* Right controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* Send button */}
          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            aria-label="Send message"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '36px',
              height: '36px',
              borderRadius: '10px',
              border: 'none',
              background: canSubmit ? '#6366f1' : '#1e1e1e',
              color: canSubmit ? '#FFFFFF' : '#444444',
              cursor: canSubmit ? 'pointer' : 'not-allowed',
              transition: 'all 0.15s ease',
              flexShrink: 0,
            }}
            onMouseEnter={(e) => {
              if (canSubmit) {
                (e.currentTarget as HTMLButtonElement).style.background = '#818CF8';
                (e.currentTarget as HTMLButtonElement).style.transform = 'scale(1.05)';
              }
            }}
            onMouseLeave={(e) => {
              if (canSubmit) {
                (e.currentTarget as HTMLButtonElement).style.background = '#6366f1';
                (e.currentTarget as HTMLButtonElement).style.transform = 'scale(1)';
              }
            }}
            onMouseDown={(e) => {
              if (canSubmit) (e.currentTarget as HTMLButtonElement).style.transform = 'scale(0.95)';
            }}
            onMouseUp={(e) => {
              if (canSubmit) (e.currentTarget as HTMLButtonElement).style.transform = 'scale(1.05)';
            }}
          >
            {submitting ? (
              <div style={{ width: '16px', height: '16px', border: '2px solid rgba(255,255,255,0.3)', borderTopColor: '#FFFFFF', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
            ) : (
              <ArrowUp size={16} strokeWidth={2.5} />
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

// ── Small control icon button ─────────────────────────────────────────────────
function ControlButton({ icon, label, onClick }: { icon: React.ReactNode; label: string; onClick: () => void }) {
  const [hov, setHov] = useState(false);
  return (
    <button
      onClick={onClick}
      aria-label={label}
      title={label}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '28px',
        height: '28px',
        borderRadius: '6px',
        border: 'none',
        background: hov ? '#1e1e1e' : 'transparent',
        color: hov ? '#FFFFFF' : '#8E8E93',
        cursor: 'pointer',
        transition: 'all 0.12s ease',
      }}
    >
      {icon}
    </button>
  );
}
