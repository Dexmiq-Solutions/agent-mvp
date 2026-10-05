import React, { useState, useEffect } from 'react';
import { X, Loader2, TriangleAlert } from 'lucide-react';
import * as projectService from '../../services/projectService';
import type { Project } from '../../types';

interface DeleteProjectModalProps {
  project: Project | null;
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (projectId: string) => void;
}

export const DeleteProjectModal: React.FC<DeleteProjectModalProps> = ({
  project,
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) setError(null);
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !loading) onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, loading, onClose]);

  if (!isOpen || !project) return null;

  const handleDelete = async () => {
    try {
      setLoading(true);
      setError(null);
      await projectService.deleteProject(project.id);
      onSuccess(project.id);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to delete project.');
    } finally {
      setLoading(false);
    }
  };

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
        background: 'rgba(0,0,0,0.7)',
        backdropFilter: 'blur(10px)',
        WebkitBackdropFilter: 'blur(10px)',
      }}
      onClick={(e) => { if (e.target === e.currentTarget && !loading) onClose(); }}
    >
      <div
        className="modal-enter"
        style={{
          width: '100%',
          maxWidth: '380px',
          background: '#111111',
          border: '1px solid #222222',
          borderRadius: '12px',
          boxShadow: '0 24px 64px rgba(0,0,0,0.8)',
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
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <TriangleAlert size={15} strokeWidth={2} color="rgba(248,113,113,0.9)" />
            <h2
              style={{
                margin: 0,
                fontSize: '15px',
                fontWeight: 600,
                letterSpacing: '-0.02em',
                color: '#f87171',
                fontFamily: 'Inter, sans-serif',
              }}
            >
              Delete Project
            </h2>
          </div>
          <ModalCloseButton onClick={onClose} disabled={loading} />
        </div>

        {/* Body */}
        <div style={{ padding: '20px' }}>
          <p
            style={{
              fontSize: '13.5px',
              color: 'rgba(148,163,184,0.8)',
              lineHeight: 1.65,
              marginBottom: '16px',
              fontFamily: 'Inter, sans-serif',
            }}
          >
            You are about to permanently delete{' '}
            <strong style={{ color: '#F1F1F3', fontWeight: 600 }}>
              "{project.name}"
            </strong>
            . This will remove all conversations and source documents associated with this project.
          </p>

          {/* Warning note */}
          <div
            style={{
              padding: '10px 13px',
              borderRadius: '8px',
              background: 'rgba(239,68,68,0.06)',
              border: '1px solid rgba(239,68,68,0.14)',
              fontSize: '12.5px',
              color: 'rgba(252,165,165,0.7)',
              fontFamily: 'Inter, sans-serif',
              lineHeight: 1.55,
              marginBottom: error ? '16px' : '20px',
            }}
          >
            This action cannot be undone.
          </div>

          {error && (
            <div
              style={{
                padding: '10px 13px',
                borderRadius: '8px',
                background: 'rgba(239,68,68,0.07)',
                border: '1px solid rgba(239,68,68,0.2)',
                fontSize: '13px',
                color: '#f87171',
                fontFamily: 'Inter, sans-serif',
                marginBottom: '20px',
              }}
            >
              {error}
            </div>
          )}

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
            <DeleteButton loading={loading} onClick={handleDelete} />
          </div>
        </div>
      </div>
    </div>
  );
};

/* ── Sub-components ── */

function ModalCloseButton({ onClick, disabled }: { onClick: () => void; disabled?: boolean }) {
  const [hov, setHov] = useState(false);
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        width: '26px', height: '26px',
        borderRadius: '6px',
        border: 'none',
        background: hov ? 'rgba(255,255,255,0.07)' : 'transparent',
        color: hov ? '#F1F1F3' : 'rgba(148,163,184,0.6)',
        cursor: disabled ? 'not-allowed' : 'pointer',
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

function DeleteButton({ loading, onClick }: { loading: boolean; onClick: () => void }) {
  const [hov, setHov] = useState(false);
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={loading}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        minWidth: '84px',
        padding: '8px 16px',
        borderRadius: '8px',
        border: '1px solid rgba(239,68,68,0.3)',
        background: loading
          ? 'rgba(239,68,68,0.08)'
          : hov ? 'rgba(239,68,68,0.18)' : 'rgba(239,68,68,0.1)',
        color: '#f87171',
        fontSize: '13px',
        fontFamily: 'Inter, sans-serif',
        fontWeight: 600,
        cursor: loading ? 'not-allowed' : 'pointer',
        transition: 'all 0.15s ease',
      }}
    >
      {loading
        ? <Loader2 size={14} strokeWidth={2} style={{ animation: 'spin 0.9s linear infinite' }} />
        : 'Delete'}
    </button>
  );
}
