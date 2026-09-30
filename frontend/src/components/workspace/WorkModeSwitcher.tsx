import React, { useState } from 'react';
import { X } from 'lucide-react';

interface WorkModeSwitcherProps {
  /** Currently not used for routing — Work mode is not yet implemented */
}

// ── Coming soon notice ────────────────────────────────────────────────────────
function ComingSoonBadge({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div
      className="slide-up"
      style={{
        position: 'absolute',
        top: 'calc(100% + 8px)',
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
      <span>Agent workflows coming soon</span>
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

export const WorkModeSwitcher: React.FC<WorkModeSwitcherProps> = () => {
  const [showWorkNotice, setShowWorkNotice] = useState(false);

  return (
    <div
      style={{
        position: 'relative',
        display: 'inline-flex',
        alignItems: 'center',
        gap: '2px',
        padding: '4px',
        background: '#141414',
        border: '1px solid #222222',
        borderRadius: '10px',
      }}
    >
      {/* Chat — active */}
      <button
        style={{
          padding: '6px 16px',
          borderRadius: '7px',
          border: '1px solid rgba(255,255,255,0.04)',
          background: '#222222',
          color: '#FFFFFF',
          fontSize: '13px',
          fontWeight: 500,
          cursor: 'default',
          fontFamily: 'Inter, sans-serif',
          letterSpacing: '-0.01em',
          boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
        }}
      >
        Chat
      </button>

      {/* Work — coming soon */}
      <button
        onClick={() => setShowWorkNotice((v) => !v)}
        style={{
          padding: '6px 16px',
          borderRadius: '7px',
          border: '1px solid transparent',
          background: 'transparent',
          color: '#8E8E93',
          fontSize: '13px',
          fontWeight: 500,
          cursor: 'pointer',
          fontFamily: 'Inter, sans-serif',
          letterSpacing: '-0.01em',
          transition: 'all 0.15s ease',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.color = '#FFFFFF';
          e.currentTarget.style.background = 'rgba(255,255,255,0.03)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.color = '#8E8E93';
          e.currentTarget.style.background = 'transparent';
        }}
      >
        + Work
      </button>

      {showWorkNotice && (
        <>
          <div
            style={{ position: 'fixed', inset: 0, zIndex: 49 }}
            onClick={() => setShowWorkNotice(false)}
          />
          <ComingSoonBadge onDismiss={() => setShowWorkNotice(false)} />
        </>
      )}
    </div>
  );
};
