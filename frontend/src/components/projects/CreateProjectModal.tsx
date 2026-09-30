import React, { useState, useEffect, useRef } from 'react';
import { X, Loader2 } from 'lucide-react';
import { mockProjectService } from '../../services/mockData';
import type { Project } from '../../types';

interface CreateProjectModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (project: Project) => void;
}

export const CreateProjectModal: React.FC<CreateProjectModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nameRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setName('');
      setDescription('');
      setError(null);
      setTimeout(() => nameRef.current?.focus(), 60);
    }
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    try {
      setLoading(true);
      setError(null);
      const project = await mockProjectService.createProject({
        name: name.trim(),
        description: description.trim() || null,
      });
      onSuccess(project);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to create project.');
    } finally {
      setLoading(false);
    }
  };

  const inputBase: React.CSSProperties = {
    width: '100%',
    padding: '8px 11px',
    borderRadius: '7px',
    border: '1px solid #2a2a2a',
    background: '#141414',
    color: '#FFFFFF',
    fontSize: '13px',
    fontFamily: 'Inter, sans-serif',
    outline: 'none',
    transition: 'border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease',
    boxSizing: 'border-box',
  };

  const focusStyle = (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    e.target.style.borderColor = 'rgba(99,102,241,0.5)';
    e.target.style.background = '#1a1a1a';
    e.target.style.boxShadow = '0 0 0 3px rgba(99,102,241,0.1)';
  };
  const blurStyle = (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    e.target.style.borderColor = '#2a2a2a';
    e.target.style.background = '#141414';
    e.target.style.boxShadow = 'none';
  };

  return (
    /* Backdrop */
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
        background: 'rgba(0,0,0,0.7)',
        backdropFilter: 'blur(10px)',
        WebkitBackdropFilter: 'blur(10px)',
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      {/* Panel */}
      <div
        className="modal-enter"
        style={{
          width: '100%',
          maxWidth: '440px',
          background: '#16181E',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: '14px',
          boxShadow: '0 24px 64px rgba(0,0,0,0.75)',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '16px 18px',
            borderBottom: '1px solid #1a1a1a',
          }}
        >
          <h2
            style={{
              margin: 0,
              fontSize: '14px',
              fontWeight: 600,
              letterSpacing: '-0.02em',
              color: '#FFFFFF',
              fontFamily: 'Inter, sans-serif',
            }}
          >
            New Project
          </h2>
          <CloseButton onClick={onClose} />
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} style={{ padding: '20px' }}>
          {error && <ErrorAlert message={error} />}

          {/* Name field */}
          <div style={{ marginBottom: '16px' }}>
            <FieldLabel text="Name" required />
            <input
              ref={nameRef}
              type="text"
              required
              maxLength={255}
              value={name}
              placeholder="e.g. Q4 Marketing Research"
              onChange={(e) => setName(e.target.value)}
              onFocus={focusStyle}
              onBlur={blurStyle}
              style={inputBase}
            />
          </div>

          {/* Description field */}
          <div style={{ marginBottom: '20px' }}>
            <FieldLabel text="Description" hint="optional" />
            <textarea
              rows={3}
              value={description}
              placeholder="What is this workspace for?"
              onChange={(e) => setDescription(e.target.value)}
              onFocus={focusStyle as any}
              onBlur={blurStyle as any}
              style={{ ...inputBase, resize: 'none', lineHeight: '1.55' }}
            />
          </div>

          {/* Actions */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'flex-end',
              gap: '8px',
              paddingTop: '16px',
              borderTop: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            <GhostButton label="Cancel" onClick={onClose} disabled={loading} />
            <PrimaryButton
              type="submit"
              label="Create"
              loading={loading}
              disabled={!name.trim() || loading}
            />
          </div>
        </form>
      </div>
    </div>
  );
};

/* ── Shared sub-components ── */

function FieldLabel({ text, required, hint }: { text: string; required?: boolean; hint?: string }) {
  return (
    <label
      style={{
        display: 'block',
        fontSize: '11.5px',
        fontWeight: 600,
        letterSpacing: '0.06em',
        textTransform: 'uppercase',
        color: 'rgba(148,163,184,0.65)',
        marginBottom: '7px',
        fontFamily: 'Inter, sans-serif',
      }}
    >
      {text}
      {required && <span style={{ color: '#f87171', marginLeft: '3px' }}>*</span>}
      {hint && (
        <span style={{ textTransform: 'none', letterSpacing: 0, fontWeight: 400, color: 'rgba(148,163,184,0.4)', marginLeft: '6px' }}>
          ({hint})
        </span>
      )}
    </label>
  );
}

function ErrorAlert({ message }: { message: string }) {
  return (
    <div
      style={{
        padding: '10px 13px',
        marginBottom: '16px',
        borderRadius: '8px',
        background: 'rgba(239,68,68,0.07)',
        border: '1px solid rgba(239,68,68,0.2)',
        fontSize: '13px',
        color: '#f87171',
        fontFamily: 'Inter, sans-serif',
      }}
    >
      {message}
    </div>
  );
}

function CloseButton({ onClick }: { onClick: () => void }) {
  const [hov, setHov] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        width: '26px', height: '26px',
        borderRadius: '6px',
        border: 'none',
        background: hov ? 'rgba(255,255,255,0.07)' : 'transparent',
        color: hov ? '#F1F1F3' : 'rgba(148,163,184,0.6)',
        cursor: 'pointer',
        transition: 'all 0.12s ease',
      }}
    >
      <X size={15} strokeWidth={2} />
    </button>
  );
}

function GhostButton({ label, onClick, disabled }: { label: string; onClick: () => void; disabled?: boolean }) {
  const [hov, setHov] = useState(false);
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        padding: '8px 14px',
        borderRadius: '8px',
        border: '1px solid rgba(255,255,255,0.09)',
        background: hov ? 'rgba(255,255,255,0.05)' : 'transparent',
        color: hov ? '#d4d4d8' : 'rgba(148,163,184,0.65)',
        fontSize: '13px',
        fontFamily: 'Inter, sans-serif',
        fontWeight: 500,
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'all 0.12s ease',
      }}
    >
      {label}
    </button>
  );
}

function PrimaryButton({
  type, label, loading, disabled,
}: {
  type?: 'submit' | 'button';
  label: string;
  loading?: boolean;
  disabled?: boolean;
}) {
  const [hov, setHov] = useState(false);
  return (
    <button
      type={type ?? 'button'}
      disabled={disabled}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        minWidth: '84px',
        padding: '8px 16px',
        borderRadius: '8px',
        border: '1px solid rgba(99,102,241,0.4)',
        background: disabled && !loading
          ? 'rgba(99,102,241,0.08)'
          : hov ? 'rgba(99,102,241,0.3)' : 'rgba(99,102,241,0.18)',
        color: disabled && !loading ? 'rgba(165,180,252,0.35)' : '#c7d2fe',
        fontSize: '13px',
        fontFamily: 'Inter, sans-serif',
        fontWeight: 600,
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'all 0.15s ease',
      }}
    >
      {loading
        ? <Loader2 size={14} strokeWidth={2} style={{ animation: 'spin 0.9s linear infinite' }} />
        : label}
    </button>
  );
}
