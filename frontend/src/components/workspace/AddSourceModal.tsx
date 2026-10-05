import React, { useState, useRef, useCallback, useEffect } from 'react';
import { X, Upload, FileText, AlertCircle } from 'lucide-react';
import * as sourceService from '../../services/sourceService';

interface AddSourceModalProps {
  isOpen: boolean;
  projectId: string;
  onClose: () => void;
  onSuccess: () => void;
}

const ACCEPTED_EXTENSIONS = ['.md', '.txt', '.docx'];
const MAX_SIZE_BYTES = 50 * 1024 * 1024; // 50 MB

function formatBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1048576).toFixed(1)} MB`;
}

function getExtension(filename: string): string {
  const idx = filename.lastIndexOf('.');
  return idx >= 0 ? filename.slice(idx).toLowerCase() : '';
}

type FileEntry = {
  file: File;
  status: 'pending' | 'uploading' | 'done' | 'error';
  error?: string;
};

export const AddSourceModal: React.FC<AddSourceModalProps> = ({
  isOpen,
  projectId,
  onClose,
  onSuccess,
}) => {
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [dragging, setDragging] = useState(false);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploading = files.some((f) => f.status === 'uploading');

  // Reset on open
  useEffect(() => {
    if (isOpen) {
      setFiles([]);
      setGlobalError(null);
    }
  }, [isOpen]);

  // Escape key
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !uploading) onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, uploading, onClose]);

  const validateAndAddFiles = useCallback((incoming: File[]) => {
    const toAdd: FileEntry[] = [];
    for (const file of incoming) {
      const ext = getExtension(file.name);

      if (ext === '.pdf') {
        toAdd.push({
          file,
          status: 'error',
          error: 'PDF processing is coming in the next update. Please upload markdown, text, or docx files for now.',
        });
        continue;
      }
      if (!ACCEPTED_EXTENSIONS.includes(ext)) {
        toAdd.push({
          file,
          status: 'error',
          error: `Unsupported file type "${ext}". Accepted: .md, .txt, .docx`,
        });
        continue;
      }
      if (file.size > MAX_SIZE_BYTES) {
        toAdd.push({
          file,
          status: 'error',
          error: `File exceeds 50 MB limit (${formatBytes(file.size)}).`,
        });
        continue;
      }
      toAdd.push({ file, status: 'pending' });
    }
    setFiles((prev) => [...prev, ...toAdd]);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const dropped = Array.from(e.dataTransfer.files);
    validateAndAddFiles(dropped);
  }, [validateAndAddFiles]);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(e.target.files ?? []);
    validateAndAddFiles(selected);
    e.target.value = '';
  };

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleUpload = async () => {
    const pendingFiles = files.filter((f) => f.status === 'pending');
    if (pendingFiles.length === 0) return;

    setGlobalError(null);

    for (let i = 0; i < files.length; i++) {
      if (files[i].status !== 'pending') continue;
      setFiles((prev) => {
        const next = [...prev];
        next[i] = { ...next[i], status: 'uploading' };
        return next;
      });
      try {
        await sourceService.uploadSource(projectId, files[i].file);
        setFiles((prev) => {
          const next = [...prev];
          next[i] = { ...next[i], status: 'done' };
          return next;
        });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Upload failed.';
        setFiles((prev) => {
          const next = [...prev];
          next[i] = { ...next[i], status: 'error', error: msg };
          return next;
        });
      }
    }

    // If any succeeded, call onSuccess after a brief moment
    setFiles((prev) => {
      const anyDone = prev.some((f) => f.status === 'done');
      if (anyDone) {
        setTimeout(onSuccess, 600);
      }
      return prev;
    });
  };

  const hasPending = files.some((f) => f.status === 'pending');
  const allDone = files.length > 0 && files.every((f) => f.status === 'done' || f.status === 'error');

  if (!isOpen) return null;

  return (
    <div
      className="backdrop-enter"
      style={{
        position: 'fixed', inset: 0, zIndex: 50,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: '16px',
        background: 'rgba(0,0,0,0.65)',
        backdropFilter: 'blur(8px)',
      }}
      onClick={(e) => { if (e.target === e.currentTarget && !uploading) onClose(); }}
    >
      <div
        className="modal-enter"
        style={{
          width: '100%', maxWidth: '460px',
          background: '#111111', border: '1px solid #222222',
          borderRadius: '14px', boxShadow: '0 24px 64px rgba(0,0,0,0.8)',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 18px', borderBottom: '1px solid #1a1a1a' }}>
          <h2 style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#FFFFFF', fontFamily: 'Inter, sans-serif' }}>
            Add sources
          </h2>
          <button
            onClick={onClose}
            disabled={uploading}
            aria-label="Close modal"
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              width: '26px', height: '26px', borderRadius: '6px',
              border: 'none', background: 'transparent', color: '#8E8E93',
              cursor: uploading ? 'not-allowed' : 'pointer',
            }}
            onMouseEnter={(e) => { if (!uploading) (e.currentTarget as HTMLButtonElement).style.color = '#FFFFFF'; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#8E8E93'; }}
          >
            <X size={14} strokeWidth={2} />
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              padding: '28px 16px', borderRadius: '10px',
              border: `1.5px dashed ${dragging ? '#6366f1' : '#2a2a2a'}`,
              background: dragging ? 'rgba(99,102,241,0.05)' : '#0A0A0A',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
              gap: '8px',
            }}
          >
            <Upload size={20} strokeWidth={1.5} color={dragging ? '#a5b4fc' : '#8E8E93'} />
            <p style={{ fontSize: '13px', color: dragging ? '#a5b4fc' : '#FFFFFF', fontWeight: 500, textAlign: 'center' }}>
              Drop files here or <span style={{ color: '#818cf8' }}>browse</span>
            </p>
            <p style={{ fontSize: '11.5px', color: '#8E8E93', textAlign: 'center' }}>
              .md, .txt, .docx — max 50 MB each
            </p>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".md,.txt,.docx"
            style={{ display: 'none' }}
            onChange={handleFileInput}
          />

          {/* Global error */}
          {globalError && (
            <div style={{ display: 'flex', gap: '8px', padding: '10px 12px', borderRadius: '8px', background: 'rgba(239,68,68,0.07)', border: '1px solid rgba(239,68,68,0.18)', fontSize: '12.5px', color: '#f87171' }}>
              <AlertCircle size={13} style={{ flexShrink: 0, marginTop: '1px' }} />
              {globalError}
            </div>
          )}

          {/* File queue */}
          {files.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '200px', overflowY: 'auto' }}>
              {files.map((entry, i) => (
                <FileQueueItem key={i} entry={entry} onRemove={() => removeFile(i)} />
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', padding: '12px 18px', borderTop: '1px solid #1a1a1a' }}>
          <button
            onClick={onClose}
            disabled={uploading}
            style={{
              padding: '7px 14px', borderRadius: '7px',
              border: '1px solid #2a2a2a', background: 'transparent',
              color: '#8E8E93', fontSize: '13px', fontFamily: 'Inter, sans-serif',
              fontWeight: 500, cursor: uploading ? 'not-allowed' : 'pointer',
            }}
          >
            Cancel
          </button>
          {allDone ? (
            <button
              onClick={onClose}
              style={{
                padding: '7px 16px', borderRadius: '7px',
                border: '1px solid rgba(99,102,241,0.35)', background: 'rgba(99,102,241,0.12)',
                color: '#a5b4fc', fontSize: '13px', fontFamily: 'Inter, sans-serif',
                fontWeight: 500, cursor: 'pointer',
              }}
            >
              Done
            </button>
          ) : (
            <button
              onClick={handleUpload}
              disabled={!hasPending || uploading}
              style={{
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                gap: '6px', minWidth: '90px', padding: '7px 16px', borderRadius: '7px',
                border: '1px solid rgba(99,102,241,0.35)',
                background: hasPending && !uploading ? 'rgba(99,102,241,0.18)' : 'rgba(99,102,241,0.06)',
                color: hasPending && !uploading ? '#a5b4fc' : 'rgba(165,180,252,0.35)',
                fontSize: '13px', fontFamily: 'Inter, sans-serif',
                fontWeight: 500, cursor: hasPending && !uploading ? 'pointer' : 'not-allowed',
              }}
            >
              {uploading && (
                <div style={{ width: '12px', height: '12px', border: '1.5px solid rgba(165,180,252,0.3)', borderTopColor: '#a5b4fc', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
              )}
              {uploading ? 'Uploading…' : 'Upload'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

// ── Single file queue entry ────────────────────────────────────────────────────
function FileQueueItem({ entry, onRemove }: { entry: FileEntry; onRemove: () => void }) {
  const ext = getExtension(entry.file.name);

  const statusColor = {
    pending:   '#8E8E93',
    uploading: '#fbbf24',
    done:      '#4ade80',
    error:     '#f87171',
  }[entry.status];

  return (
    <div
      style={{
        display: 'flex', alignItems: 'flex-start', gap: '8px',
        padding: '8px 10px', borderRadius: '8px',
        background: '#141414', border: '1px solid #1e1e1e',
      }}
    >
      <FileText size={13} strokeWidth={1.75} color="#8E8E93" style={{ flexShrink: 0, marginTop: '2px' }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{ fontSize: '12.5px', color: '#FFFFFF', fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {entry.file.name}
        </p>
        {entry.status === 'error' && entry.error ? (
          <p style={{ fontSize: '11px', color: '#f87171', lineHeight: 1.4, marginTop: '2px' }}>{entry.error}</p>
        ) : (
          <p style={{ fontSize: '11px', color: '#8E8E93', marginTop: '1px' }}>
            {ext} · {formatBytes(entry.file.size)}
          </p>
        )}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
        {entry.status === 'uploading' && (
          <div style={{ width: '12px', height: '12px', border: '1.5px solid rgba(251,191,36,0.3)', borderTopColor: '#fbbf24', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
        )}
        {entry.status !== 'uploading' && (
          <span style={{ fontSize: '10.5px', fontWeight: 500, color: statusColor, textTransform: 'capitalize' }}>
            {entry.status === 'pending' ? 'Ready' : entry.status}
          </span>
        )}
        {entry.status !== 'uploading' && entry.status !== 'done' && (
          <button
            onClick={onRemove}
            aria-label="Remove file"
            style={{ display: 'flex', alignItems: 'center', background: 'none', border: 'none', color: '#8E8E93', cursor: 'pointer', padding: 0 }}
            onMouseEnter={(e) => (e.currentTarget as HTMLButtonElement).style.color = '#FFFFFF'}
            onMouseLeave={(e) => (e.currentTarget as HTMLButtonElement).style.color = '#8E8E93'}
          >
            <X size={12} strokeWidth={2} />
          </button>
        )}
      </div>
    </div>
  );
}
