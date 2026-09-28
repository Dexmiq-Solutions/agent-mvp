import React, { useState, useRef } from 'react';
import { SourceDocument } from '../types';

interface SourceListProps {
  sources: SourceDocument[];
  loading: boolean;
  uploading: boolean;
  error: string | null;
  onUploadSource: (file: File, name?: string) => Promise<void>;
  onDeleteSource: (documentId: string) => Promise<void>;
  onRefresh: () => void;
}

export const SourceList: React.FC<SourceListProps> = ({
  sources,
  loading,
  uploading,
  error,
  onUploadSource,
  onDeleteSource,
  onRefresh,
}) => {
  const [showUploadForm, setShowUploadForm] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [customName, setCustomName] = useState('');
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      if (!customName) {
        setCustomName(file.name);
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setUploadError('Please choose a file to upload');
      return;
    }

    setUploadError(null);
    try {
      await onUploadSource(selectedFile, customName.trim() || undefined);
      setSelectedFile(null);
      setCustomName('');
      setShowUploadForm(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed');
    }
  };

  const handleDeleteClick = async (sourceId: string, sourceName: string) => {
    if (
      !window.confirm(
        `Are you sure you want to delete "${sourceName}"?\n\nThis will remove the document and all associated vector embeddings from the knowledge base.`
      )
    ) {
      return;
    }

    setDeletingId(sourceId);
    setUploadError(null);
    try {
      await onDeleteSource(sourceId);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Failed to delete source');
    } finally {
      setDeletingId(null);
    }
  };

  const getStatusBadge = (status?: string) => {
    switch (status?.toLowerCase()) {
      case 'ready':
        return <span className="badge badge-success">ready</span>;
      case 'indexing':
        return <span className="badge badge-warning">indexing</span>;
      case 'failed':
        return <span className="badge badge-danger">failed</span>;
      default:
        return <span className="badge badge-neutral">{status || 'pending'}</span>;
    }
  };

  return (
    <div className="workspace-panel source-panel" id="source-panel">
      <div className="panel-header">
        <div className="panel-title-group">
          <h3>Sources</h3>
          <span className="count-badge">{sources.length}</span>
        </div>
        <div className="panel-actions">
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={onRefresh}
            disabled={loading}
            title="Refresh sources"
            id="btn-refresh-sources"
          >
            ↻
          </button>
          {!showUploadForm && (
            <button
              type="button"
              className="btn btn-sm btn-primary"
              onClick={() => setShowUploadForm(true)}
              id="btn-upload-source"
            >
              [ Upload Source ]
            </button>
          )}
        </div>
      </div>

      {(error || uploadError) && (
        <div className="alert alert-error" id="sources-error-banner">
          {uploadError || error}
        </div>
      )}

      {showUploadForm && (
        <div className="upload-box card" id="upload-source-card">
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="source-file-input">Select File (.txt, .md, .docx, .pdf) *</label>
              <input
                ref={fileInputRef}
                id="source-file-input"
                type="file"
                onChange={handleFileChange}
                disabled={uploading}
                required
              />
            </div>
            {selectedFile && (
              <div className="form-group">
                <label htmlFor="source-custom-name">Display Name (Optional)</label>
                <input
                  id="source-custom-name"
                  type="text"
                  placeholder="Document display name"
                  value={customName}
                  onChange={(e) => setCustomName(e.target.value)}
                  disabled={uploading}
                />
              </div>
            )}
            <div className="inline-form-actions">
              <button
                type="submit"
                className="btn btn-sm btn-primary"
                disabled={uploading || !selectedFile}
                id="btn-submit-upload-source"
              >
                {uploading ? 'Uploading...' : 'Confirm Upload'}
              </button>
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                onClick={() => {
                  setShowUploadForm(false);
                  setSelectedFile(null);
                  setCustomName('');
                  setUploadError(null);
                  if (fileInputRef.current) fileInputRef.current.value = '';
                }}
                disabled={uploading}
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {loading && sources.length === 0 ? (
        <div className="panel-loading">Loading sources...</div>
      ) : sources.length === 0 ? (
        <div className="panel-empty" id="sources-empty">
          <p>No sources uploaded yet.</p>
          <p className="subtext">Uploaded sources become the knowledge base for RAG.</p>
        </div>
      ) : (
        <ul className="item-list" id="sources-list">
          {sources.map((source) => {
            const latestVer = source.latest_version;
            return (
              <li key={source.id} className="list-item" id={`source-item-${source.id}`}>
                <div className="item-main">
                  <div className="item-title" title={source.name}>
                    📄 {source.name}
                  </div>
                  {latestVer && latestVer.original_filename !== source.name && (
                    <div className="item-sub">{latestVer.original_filename}</div>
                  )}
                </div>
                <div className="item-meta">
                  {latestVer ? getStatusBadge(latestVer.status) : null}
                  {latestVer?.size_bytes ? (
                    <span className="file-size">
                      {(latestVer.size_bytes / 1024).toFixed(1)} KB
                    </span>
                  ) : null}
                  <button
                    type="button"
                    className="btn btn-sm btn-delete-source"
                    onClick={() => handleDeleteClick(source.id, source.name)}
                    disabled={deletingId === source.id || uploading}
                    title={`Delete source "${source.name}"`}
                    id={`btn-delete-source-${source.id}`}
                  >
                    {deletingId === source.id ? '...' : '✕'}
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};
