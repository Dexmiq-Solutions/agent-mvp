import React, { useState, useRef, useEffect } from 'react';
import { Folder, Share2, MoreVertical, Trash2, X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import type { Project } from '../../types';

interface ProjectHeaderProps {
  project: Project;
  onDeleteRequest: () => void;
}

// ── Toast for "coming soon" notices ─────────────────────────────────────────
function ComingSoonToast({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  useEffect(() => {
    const t = setTimeout(onDismiss, 3000);
    return () => clearTimeout(t);
  }, [onDismiss]);
  return (
    <div
      className="slide-up"
      style={{
        position: 'fixed',
        bottom: '24px',
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        padding: '10px 16px',
        background: '#1a1a1a',
        border: '1px solid #2e2e2e',
        borderRadius: '8px',
        boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        fontSize: '13px',
        color: '#FFFFFF',
        whiteSpace: 'nowrap',
      }}
    >
      <span>{message}</span>
      <button
        onClick={onDismiss}
        aria-label="Dismiss"
        style={{ background: 'none', border: 'none', color: '#8E8E93', cursor: 'pointer', display: 'flex', padding: 0 }}
      >
        <X size={13} />
      </button>
    </div>
  );
}

export const ProjectHeader: React.FC<ProjectHeaderProps> = ({ project, onDeleteRequest }) => {
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu on outside click
  useEffect(() => {
    if (!menuOpen) return;
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [menuOpen]);

  const showToast = (msg: string) => {
    setToast(msg);
    setMenuOpen(false);
  };

  return (
    <>
      <header
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 24px',
          height: '60px',
          flexShrink: 0,
          background: '#0A0A0A',
          borderBottom: '1px solid #222222',
          gap: '12px',
        }}
      >
        {/* Left: Folder + Project name */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', minWidth: 0 }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              background: '#141414',
              border: '1px solid #222222',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}
          >
            <Folder size={18} strokeWidth={1.5} color="rgba(99,102,241,0.8)" />
          </div>
          <div style={{ minWidth: 0 }}>
            <h1
              style={{
                fontSize: '16px',
                fontWeight: 600,
                color: '#FFFFFF',
                letterSpacing: '-0.01em',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                margin: 0,
              }}
            >
              {project.name}
            </h1>
          </div>
        </div>

        {/* Right: Share + overflow */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexShrink: 0 }}>
          <HeaderButton
            icon={<Share2 size={14} strokeWidth={1.75} />}
            label="Share project"
            onClick={() => showToast('Sharing is coming soon')}
          />

          {/* Overflow menu */}
          <div style={{ position: 'relative' }} ref={menuRef}>
            <HeaderButton
              icon={<MoreVertical size={14} strokeWidth={1.75} />}
              label="More options"
              onClick={() => setMenuOpen((v) => !v)}
              active={menuOpen}
            />
            {menuOpen && (
              <>
                <div
                  style={{ position: 'fixed', inset: 0, zIndex: 30 }}
                  onClick={() => setMenuOpen(false)}
                />
                <div
                  className="modal-enter"
                  style={{
                    position: 'absolute',
                    right: 0,
                    top: 'calc(100% + 6px)',
                    minWidth: '180px',
                    background: '#111111',
                    border: '1px solid #2a2a2a',
                    borderRadius: '9px',
                    padding: '4px',
                    boxShadow: '0 12px 40px rgba(0,0,0,0.6)',
                    zIndex: 40,
                  }}
                >
                  <OverflowItem
                    label="Back to projects"
                    onClick={() => { navigate('/projects'); setMenuOpen(false); }}
                  />
                  <div style={{ height: '1px', background: '#1e1e1e', margin: '3px 0' }} />
                  <OverflowItem
                    label="Delete workspace"
                    danger
                    onClick={() => { onDeleteRequest(); setMenuOpen(false); }}
                  />
                </div>
              </>
            )}
          </div>
        </div>
      </header>

      {toast && <ComingSoonToast message={toast} onDismiss={() => setToast(null)} />}
    </>
  );
};

// ── Small header icon button ──────────────────────────────────────────────────
function HeaderButton({
  icon, label, onClick, active = false,
}: {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  active?: boolean;
}) {
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
        width: '30px',
        height: '30px',
        borderRadius: '7px',
        border: active ? '1px solid #2a2a2a' : '1px solid transparent',
        background: active || hov ? '#1a1a1a' : 'transparent',
        color: hov || active ? '#FFFFFF' : '#8E8E93',
        cursor: 'pointer',
        transition: 'all 0.12s ease',
      }}
    >
      {icon}
    </button>
  );
}

// ── Overflow dropdown item ────────────────────────────────────────────────────
function OverflowItem({
  label, onClick, danger = false,
}: {
  label: string;
  onClick: () => void;
  danger?: boolean;
}) {
  const [hov, setHov] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        width: '100%',
        padding: '7px 10px',
        borderRadius: '6px',
        border: 'none',
        background: hov ? (danger ? 'rgba(239,68,68,0.09)' : 'rgba(255,255,255,0.05)') : 'transparent',
        color: danger ? (hov ? '#f87171' : 'rgba(252,165,165,0.75)') : (hov ? '#FFFFFF' : 'rgba(255,255,255,0.6)'),
        fontSize: '12.5px',
        fontFamily: 'Inter, sans-serif',
        fontWeight: 500,
        cursor: 'pointer',
        transition: 'all 0.1s ease',
        textAlign: 'left',
      }}
    >
      {danger && <Trash2 size={12} strokeWidth={2} style={{ flexShrink: 0 }} />}
      {label}
    </button>
  );
}
