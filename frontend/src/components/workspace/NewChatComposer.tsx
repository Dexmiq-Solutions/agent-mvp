import React, { useState, useRef, useCallback } from 'react';
import { ArrowUp } from 'lucide-react';

interface NewChatComposerProps {
  projectName: string;
  onSubmit: (prompt: string) => Promise<void>;
}



export const NewChatComposer: React.FC<NewChatComposerProps> = ({ projectName, onSubmit }) => {
  const [input, setInput] = useState('');
  const [submitting, setSubmitting] = useState(false);
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
      className="group relative flex flex-col overflow-hidden rounded-[24px] transition-all duration-300
                 bg-zinc-950/60 backdrop-blur-xl border border-white/10
                 shadow-2xl shadow-black/40
                 focus-within:border-indigo-500/40 focus-within:bg-zinc-900/80 focus-within:shadow-[0_0_30px_rgba(99,102,241,0.15)]"
    >
      {/* Subtle multi-color gradient background glow behind the composer */}
      <div className="pointer-events-none absolute inset-0 -z-10 opacity-0 group-focus-within:opacity-100 transition-opacity duration-500 bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-cyan-500/10" />
      {/* Textarea */}
      <div style={{ padding: '20px 24px 12px', flex: 1 }}>
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
            fontSize: '15.5px',
            fontFamily: 'Inter, sans-serif',
            lineHeight: '1.6',
            minHeight: '52px',
            maxHeight: '240px',
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
        <div className="flex items-center gap-2">
          {/* Controls can go here */}
        </div>

        {/* Right controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            aria-label="Send message"
            className={`
              flex items-center justify-center w-9 h-9 rounded-xl shrink-0 transition-all duration-200
              ${canSubmit
                ? 'bg-gradient-to-tr from-indigo-600 to-indigo-500 text-white shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/50 hover:-translate-y-0.5 active:translate-y-0'
                : 'bg-white/5 text-zinc-600 cursor-not-allowed'
              }
            `}
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

