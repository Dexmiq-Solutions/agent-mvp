import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
  FileText, AlertCircle, RotateCcw, CheckCircle2,
  AlertTriangle, Pencil, Trash2, Upload, Plus,
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { mockSourceService } from '../../services/mockData';
import type { SourceDocument, DocumentVersionStatus } from '../../types';
import { AddSourceModal } from './AddSourceModal';

interface SourcesTabProps {
  projectId: string;
}

// ── Source skeleton ───────────────────────────────────────────────────────────
function SourceSkeleton() {
  return (
    <div
      className="skeleton"
      style={{ height: '68px', borderRadius: '10px', flexShrink: 0 }}
    />
  );
}

// ── Status badge ──────────────────────────────────────────────────────────────
function StatusBadge({ status }: { status: DocumentVersionStatus | undefined }) {
  if (!status) return null;

  const configs: Record<DocumentVersionStatus, { label: string; color: string; bg: string; border: string; spin?: boolean }> = {
    pending:  { label: 'Processing…', color: '#fbbf24', bg: 'rgba(251,191,36,0.08)',   border: 'rgba(251,191,36,0.2)', spin: true },
    indexing: { label: 'Processing…', color: '#fbbf24', bg: 'rgba(251,191,36,0.08)',   border: 'rgba(251,191,36,0.2)', spin: true },
    ready:    { label: 'Ready',       color: '#4ade80', bg: 'rgba(74,222,128,0.08)',    border: 'rgba(74,222,128,0.2)' },
    failed:   { label: 'Failed',      color: '#f87171', bg: 'rgba(239,68,68,0.08)',     border: 'rgba(239,68,68,0.2)' },
  };
  const cfg = configs[status];

  return (
    <span
      style={{
        display: 'inline-flex', alignItems: 'center', gap: '5px',
        padding: '2px 7px', borderRadius: '5px',
        background: cfg.bg, border: `1px solid ${cfg.border}`,
        fontSize: '11px', color: cfg.color, fontWeight: 500, flexShrink: 0,
      }}
    >
      {cfg.spin && (
        <div style={{ width: '8px', height: '8px', border: '1.5px solid rgba(251,191,36,0.3)', borderTopColor: '#fbbf24', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
      )}
      {!cfg.spin && status === 'ready' && <CheckCircle2 size={10} strokeWidth={2} />}
      {!cfg.spin && status === 'failed' && <AlertTriangle size={10} strokeWidth={2} />}
      {cfg.label}
    </span>
  );
}

// ── Format bytes helper ───────────────────────────────────────────────────────
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

export const SourcesTab: React.FC<SourcesTabProps> = ({ projectId }) => {
  const [sources, setSources] = useState<SourceDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [_pendingDelete, setPendingDelete] = useState<SourceDocument | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchSources = useCallback(async (silent = false) => {
    try {
      if (!silent) { setError(null); setLoading(true); }
      const res = await mockSourceService.listSources(projectId, 50, 0);
      setSources(res.items);
    } catch (err: unknown) {
      if (!silent) setError(err instanceof Error ? err.message : 'Failed to load sources.');
    } finally {
      if (!silent) setLoading(false);
    }
  }, [projectId]);

  // Initial fetch
  useEffect(() => { fetchSources(); }, [fetchSources]);

  // Polling for pending/indexing sources
  useEffect(() => {
    const hasPending = sources.some(
      (s) => s.latest_version?.status === 'pending' || s.latest_version?.status === 'indexing'
    );

    if (hasPending) {
      if (!pollingRef.current) {
        pollingRef.current = setInterval(() => fetchSources(true), 2000);
      }
    } else {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [sources, fetchSources]);

  const handleDelete = async (source: SourceDocument) => {
    if (!window.confirm(`Delete "${source.name}"? This cannot be undone.`)) return;
    try {
      await mockSourceService.deleteSource(projectId, source.id);
      setSources((prev) => prev.filter((s) => s.id !== source.id));
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to delete source.');
    }
    setPendingDelete(null);
  };

  const handleRename = async (source: SourceDocument, newName: string) => {
    if (!newName.trim() || newName.trim() === source.name) return;
    try {
      const updated = await mockSourceService.updateSource(projectId, source.id, { name: newName.trim() });
      setSources((prev) => prev.map((s) => (s.id === updated.id ? updated : s)));
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to rename source.');
    }
    setRenamingId(null);
  };

  const handleUploadVersion = async (source: SourceDocument, file: File) => {
    try {
      await mockSourceService.uploadSourceVersion(projectId, source.id, file);
      await fetchSources(true);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to upload new version.');
    }
  };

  if (loading) {
    return (
      <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {[0, 1, 2, 3].map((i) => <SourceSkeleton key={i} />)}
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '24px 16px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px', textAlign: 'center' }}>
        <AlertCircle size={20} color="#f87171" strokeWidth={1.75} />
        <p style={{ fontSize: '13px', color: '#8E8E93' }}>Failed to load sources.</p>
        <button
          onClick={() => fetchSources()}
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

  return (
    <>
      <div style={{ paddingTop: '24px', display: 'flex', flexDirection: 'column' }}>
        {/* Toolbar */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', marginBottom: '16px' }}>
          <button
            onClick={() => setAddOpen(true)}
            style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              padding: '7px 14px', borderRadius: '8px',
              border: '1px solid rgba(255,255,255,0.1)',
              background: '#1a1a1a', color: '#FFFFFF',
              fontSize: '13px', fontWeight: 500, cursor: 'pointer',
              transition: 'all 0.15s ease', boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLButtonElement).style.background = '#222222';
              (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.2)';
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLButtonElement).style.background = '#1a1a1a';
              (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.1)';
            }}
          >
            <Plus size={14} strokeWidth={2} />
            Add sources
          </button>
        </div>

        {/* Source list */}
        {sources.length === 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', paddingTop: '40px', textAlign: 'center' }}>
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5 relative
                         bg-gradient-to-b from-zinc-800/50 to-zinc-900/50 backdrop-blur-md border border-white/10
                         shadow-[inset_0_1px_0_rgba(255,255,255,0.1)]"
            >
              <div className="absolute inset-0 bg-indigo-500/20 blur-xl rounded-full" />
              <FileText size={26} strokeWidth={1.5} className="text-zinc-400 relative z-10" />
            </div>
            <p className="text-[15px] font-semibold text-white mb-2">No knowledge base documents yet</p>
            <p className="text-[13.5px] text-zinc-400 max-w-[300px] leading-relaxed mb-6">
              Add documents to ground your AI assistant in this project.
            </p>
            <button
              onClick={() => setAddOpen(true)}
              style={{
                display: 'flex', alignItems: 'center', gap: '6px',
                padding: '7px 14px', borderRadius: '8px',
                border: '1px solid rgba(255,255,255,0.1)',
                background: '#1a1a1a', color: '#FFFFFF',
                fontSize: '13px', fontWeight: 500, cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLButtonElement).style.background = '#222222';
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLButtonElement).style.background = '#1a1a1a';
              }}
            >
              <Plus size={14} strokeWidth={2} />
              Add sources
            </button>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {sources.map((source) => (
              <SourceCard
                key={source.id}
                source={source}
                isRenaming={renamingId === source.id}
                onRenameStart={() => setRenamingId(source.id)}
                onRenameSubmit={(name) => handleRename(source, name)}
                onRenameCancel={() => setRenamingId(null)}
                onDeleteRequest={() => handleDelete(source)}
                onUploadVersion={(file) => handleUploadVersion(source, file)}
              />
            ))}
          </div>
        )}
      </div>

      <AddSourceModal
        isOpen={addOpen}
        projectId={projectId}
        onClose={() => setAddOpen(false)}
        onSuccess={() => {
          setAddOpen(false);
          fetchSources(true);
        }}
      />
    </>
  );
};

// ── Source card ───────────────────────────────────────────────────────────────
function SourceCard({
  source,
  isRenaming,
  onRenameStart,
  onRenameSubmit,
  onRenameCancel,
  onDeleteRequest,
  onUploadVersion,
}: {
  source: SourceDocument;
  isRenaming: boolean;
  onRenameStart: () => void;
  onRenameSubmit: (name: string) => void;
  onRenameCancel: () => void;
  onDeleteRequest: () => void;
  onUploadVersion: (file: File) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const [renameValue, setRenameValue] = useState(source.name);
  const renameInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const status = source.latest_version?.status;
  const version = source.latest_version;

  useEffect(() => {
    if (isRenaming) {
      setRenameValue(source.name);
      setTimeout(() => renameInputRef.current?.select(), 30);
    }
  }, [isRenaming, source.name]);

  const updatedAt = source.updated_at
    ? formatDistanceToNow(new Date(source.updated_at), { addSuffix: true })
    : null;

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        padding: '18px 22px',
        borderRadius: '14px',
        border: hovered ? '1px solid rgba(255,255,255,0.15)' : '1px solid #1e1e1e',
        background: hovered ? '#171717' : '#111111',
        transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
        gap: '12px',
        flexWrap: 'wrap',
        boxShadow: hovered ? '0 4px 12px rgba(0,0,0,0.2)' : 'none',
        transform: hovered ? 'translateY(-1px)' : 'translateY(0)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px', minWidth: 0, flex: 1 }}>
        {/* Icon */}
        <div
          style={{
            width: '40px', height: '40px', borderRadius: '12px', flexShrink: 0,
            background: hovered ? 'rgba(99,102,241,0.08)' : '#171717',
            border: hovered ? '1px solid rgba(99,102,241,0.2)' : '1px solid #222',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'all 0.15s ease',
          }}
        >
          <FileText
            size={18}
            strokeWidth={1.5}
            color={hovered ? '#a5b4fc' : '#8E8E93'}
            style={{ transition: 'color 0.15s ease' }}
          />
        </div>

        {/* Content */}
        <div style={{ minWidth: 0, flex: 1 }}>
          {isRenaming ? (
            <input
              ref={renameInputRef}
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') onRenameSubmit(renameValue);
                if (e.key === 'Escape') onRenameCancel();
              }}
              onBlur={() => onRenameSubmit(renameValue)}
              style={{
                width: '100%', background: '#1a1a1a', border: '1px solid rgba(99,102,241,0.4)',
                borderRadius: '6px', padding: '4px 8px', color: '#FFFFFF',
                fontSize: '14.5px', fontFamily: 'Inter, sans-serif', outline: 'none',
                boxShadow: '0 0 0 3px rgba(99,102,241,0.1)',
                marginBottom: '4px',
              }}
            />
          ) : (
            <p
              style={{
                fontSize: '14.5px', fontWeight: 500, color: '#FFFFFF',
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                marginBottom: '4px', letterSpacing: '-0.01em',
              }}
            >
              {source.name}
            </p>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '12px', color: '#8E8E93' }}>File</span>
            {version?.size_bytes != null && (
              <>
                <span style={{ fontSize: '12px', color: '#444' }}>·</span>
                <span style={{ fontSize: '12px', color: '#8E8E93' }}>
                  {formatBytes(version.size_bytes)}
                </span>
              </>
            )}
            {updatedAt && (
              <>
                <span style={{ fontSize: '12px', color: '#444' }}>·</span>
                <span style={{ fontSize: '12px', color: '#8E8E93' }}>{updatedAt}</span>
              </>
            )}
            {status && (
              <>
                <span style={{ fontSize: '12px', color: '#444' }}>·</span>
                <StatusBadge status={status} />
              </>
            )}
            {status === 'failed' && version?.error_message && (
              <span style={{ fontSize: '11px', color: '#f87171', maxWidth: '200px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {version.error_message}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Actions (visible on hover) */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '2px',
          opacity: hovered ? 1 : 0,
          transition: 'opacity 0.12s ease',
          flexShrink: 0,
        }}
      >
        <SourceActionButton
          icon={<Pencil size={12} strokeWidth={2} />}
          label="Rename source"
          onClick={onRenameStart}
        />
        <SourceActionButton
          icon={<Upload size={12} strokeWidth={2} />}
          label="Upload new version"
          onClick={() => fileInputRef.current?.click()}
        />
        <SourceActionButton
          icon={<Trash2 size={12} strokeWidth={2} />}
          label="Delete source"
          danger
          onClick={onDeleteRequest}
        />
        {/* Hidden file input for version upload */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".md,.txt,.docx"
          style={{ display: 'none' }}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) {
              onUploadVersion(file);
              e.target.value = '';
            }
          }}
        />
      </div>
    </div>
  );
}

// ── Small source action button ─────────────────────────────────────────────────
function SourceActionButton({
  icon, label, onClick, danger = false,
}: {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  danger?: boolean;
}) {
  const [hov, setHov] = useState(false);
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onClick(); }}
      aria-label={label}
      title={label}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        width: '32px', height: '32px', borderRadius: '8px',
        border: hov ? (danger ? '1px solid rgba(239,68,68,0.2)' : '1px solid #2a2a2a') : '1px solid transparent',
        background: hov ? (danger ? 'rgba(239,68,68,0.1)' : '#222222') : 'transparent',
        color: danger ? (hov ? '#f87171' : 'rgba(252,165,165,0.7)') : (hov ? '#FFFFFF' : '#8E8E93'),
        cursor: 'pointer',
        transition: 'all 0.15s ease',
      }}
    >
      {icon}
    </button>
  );
}
