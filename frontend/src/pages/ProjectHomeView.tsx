import React, { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { AlertCircle, RotateCcw, FolderOpen } from 'lucide-react';
import * as projectService from '../services/projectService';
import { createConversation } from '../services/chatService';
import type { Project } from '../types';
import { ProjectHeader } from '../components/workspace/ProjectHeader';
import { NewChatComposer } from '../components/workspace/NewChatComposer';
import { WorkspaceTabs } from '../components/workspace/WorkspaceTabs';
import { ChatsTab } from '../components/workspace/ChatsTab';
import { SourcesTab } from '../components/workspace/SourcesTab';

// ── Header skeleton (matches ProjectHeader height) ────────────────────────────
function HeaderSkeleton() {
  return (
    <div
      style={{
        height: '52px',
        flexShrink: 0,
        borderBottom: '1px solid #1e1e1e',
        display: 'flex',
        alignItems: 'center',
        padding: '0 24px',
        gap: '10px',
      }}
    >
      <div className="skeleton" style={{ width: '28px', height: '28px', borderRadius: '7px' }} />
      <div className="skeleton" style={{ width: '160px', height: '14px', borderRadius: '5px' }} />
    </div>
  );
}

// ── Composer/tab skeleton ─────────────────────────────────────────────────────
function WorkspaceSkeleton() {
  return (
    <div style={{ padding: '24px 24px 20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Composer skeleton */}
      <div className="skeleton" style={{ height: '110px', borderRadius: '14px' }} />
      {/* Tab bar skeleton */}
      <div style={{ display: 'flex', gap: '16px', paddingBottom: '10px', borderBottom: '1px solid #1e1e1e' }}>
        <div className="skeleton" style={{ width: '50px', height: '12px', borderRadius: '5px', border: 'none' }} />
        <div className="skeleton" style={{ width: '55px', height: '12px', borderRadius: '5px', border: 'none' }} />
      </div>
      {/* Card skeletons */}
      {[0, 1, 2].map((i) => (
        <div key={i} className="skeleton" style={{ height: '68px', borderRadius: '10px' }} />
      ))}
    </div>
  );
}

// ── 404 Not Found state ───────────────────────────────────────────────────────
function NotFound() {
  const navigate = useNavigate();
  return (
    <div
      style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        padding: '48px 24px', textAlign: 'center',
      }}
    >
      <div
        style={{
          width: '52px', height: '52px', borderRadius: '13px',
          background: '#141414', border: '1px solid #222222',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          marginBottom: '18px',
        }}
      >
        <FolderOpen size={22} strokeWidth={1.5} color="#8E8E93" />
      </div>
      <h1 style={{ fontSize: '17px', fontWeight: 700, color: '#FFFFFF', marginBottom: '6px', letterSpacing: '-0.02em' }}>
        Workspace Not Found
      </h1>
      <p style={{ fontSize: '13px', color: '#8E8E93', maxWidth: '260px', lineHeight: 1.6, marginBottom: '20px' }}>
        The requested project could not be found.
      </p>
      <button
        onClick={() => navigate('/projects')}
        style={{
          padding: '7px 16px', borderRadius: '7px',
          border: '1px solid rgba(99,102,241,0.35)',
          background: 'rgba(99,102,241,0.12)',
          color: '#a5b4fc', fontSize: '13px',
          fontFamily: 'Inter, sans-serif', fontWeight: 500, cursor: 'pointer',
        }}
      >
        Return to Projects
      </button>
    </div>
  );
}

// ── General error state ───────────────────────────────────────────────────────
function ErrorState({ message: _message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div
      style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        padding: '48px 24px', textAlign: 'center',
      }}
    >
      <div
        style={{
          display: 'flex', alignItems: 'flex-start', gap: '10px',
          padding: '14px 16px', borderRadius: '9px',
          background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.15)',
          maxWidth: '360px', textAlign: 'left',
        }}
      >
        <AlertCircle size={14} strokeWidth={2} color="#f87171" style={{ flexShrink: 0, marginTop: '1px' }} />
        <div style={{ flex: 1 }}>
          <p style={{ fontSize: '13px', fontWeight: 500, color: '#f87171', marginBottom: '2px' }}>
            Failed to load workspace
          </p>
          <p style={{ fontSize: '12.5px', color: 'rgba(252,165,165,0.65)' }}>
            Please check your connection and try again.
          </p>
        </div>
        <button
          onClick={onRetry}
          aria-label="Retry loading workspace"
          style={{
            display: 'flex', alignItems: 'center', gap: '5px',
            padding: '5px 10px', borderRadius: '6px',
            border: '1px solid rgba(239,68,68,0.2)',
            background: 'rgba(239,68,68,0.08)',
            color: 'rgba(252,165,165,0.8)',
            fontSize: '12px', fontFamily: 'Inter, sans-serif',
            fontWeight: 500, cursor: 'pointer', flexShrink: 0,
          }}
        >
          <RotateCcw size={11} strokeWidth={2} />
          Retry
        </button>
      </div>
    </div>
  );
}

// ── Delete project modal ──────────────────────────────────────────────────────
function DeleteProjectConfirm({
  project,
  onConfirm,
  onCancel,
  deleting,
}: {
  project: Project;
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
      className="backdrop-enter"
      style={{
        position: 'fixed', inset: 0, zIndex: 50,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: '16px', background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(8px)',
      }}
      onClick={(e) => { if (e.target === e.currentTarget && !deleting) onCancel(); }}
    >
      <div
        className="modal-enter"
        style={{
          width: '100%', maxWidth: '360px',
          background: '#111111', border: '1px solid #222222',
          borderRadius: '12px', boxShadow: '0 24px 64px rgba(0,0,0,0.8)',
          padding: '20px',
        }}
      >
        <p style={{ fontSize: '14px', color: '#FFFFFF', fontWeight: 600, marginBottom: '6px' }}>
          Delete workspace?
        </p>
        <p style={{ fontSize: '13px', color: '#8E8E93', lineHeight: 1.55, marginBottom: '6px' }}>
          <span style={{ color: 'rgba(255,255,255,0.7)', fontWeight: 500 }}>"{project.name}"</span> and all its conversations and sources will be permanently deleted.
        </p>
        <p style={{ fontSize: '12px', color: 'rgba(252,165,165,0.7)', marginBottom: '20px' }}>
          This action cannot be undone.
        </p>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
          <button
            onClick={onCancel}
            disabled={deleting}
            style={{
              padding: '7px 14px', borderRadius: '7px',
              border: '1px solid #2a2a2a', background: 'transparent',
              color: '#8E8E93', fontSize: '13px', fontFamily: 'Inter, sans-serif',
              fontWeight: 500, cursor: deleting ? 'not-allowed' : 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={deleting}
            style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              minWidth: '80px', padding: '7px 14px', borderRadius: '7px',
              border: '1px solid rgba(239,68,68,0.3)', background: 'rgba(239,68,68,0.1)',
              color: '#f87171', fontSize: '13px', fontFamily: 'Inter, sans-serif',
              fontWeight: 600, cursor: deleting ? 'not-allowed' : 'pointer',
            }}
          >
            {deleting
              ? <div style={{ width: '14px', height: '14px', border: '2px solid rgba(248,113,113,0.3)', borderTopColor: '#f87171', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
              : 'Delete'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main ProjectHomeView ──────────────────────────────────────────────────────
type LoadState = 'loading' | 'not_found' | 'error' | 'loaded';

export const ProjectHomeView: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [project, setProject] = useState<Project | null>(null);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const fetchProject = useCallback(async () => {
    if (!projectId) { setLoadState('not_found'); return; }
    try {
      setLoadState('loading');
      setErrorMsg(null);
      const p = await projectService.getProject(projectId);
      setProject(p);
      setLoadState('loaded');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load project.';
      // Detect 404
      if (msg.includes('404') || msg.toLowerCase().includes('not found')) {
        setLoadState('not_found');
      } else {
        setErrorMsg(msg);
        setLoadState('error');
      }
    }
  }, [projectId]);

  useEffect(() => { fetchProject(); }, [fetchProject]);

  // New chat submission
  const handleNewChat = useCallback(async (prompt: string) => {
    if (!projectId) return;
    // Derive title from first 60 chars of the prompt
    const title = prompt.length > 60 ? prompt.slice(0, 60).trimEnd() + '…' : prompt;
    const conversation = await createConversation(projectId, { title });
    navigate(`/projects/${projectId}/c/${conversation.id}`, {
      state: { initialPrompt: prompt },
    });
  }, [projectId, navigate]);

  // Delete workspace
  const handleDeleteProject = async () => {
    if (!project) return;
    try {
      setDeleting(true);
      await projectService.deleteProject(project.id);
      navigate('/projects');
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to delete workspace.');
      setDeleting(false);
    }
    setPendingDelete(false);
  };

  // ── Render states ──
  if (loadState === 'not_found') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, height: '100%', background: '#0A0A0A' }}>
        <NotFound />
      </div>
    );
  }

  if (loadState === 'error') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, height: '100%', background: '#0A0A0A' }}>
        <ErrorState message={errorMsg ?? ''} onRetry={fetchProject} />
      </div>
    );
  }

  if (loadState === 'loading' || !project) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, height: '100%', background: '#0A0A0A', overflowY: 'auto' }}>
        <HeaderSkeleton />
        <WorkspaceSkeleton />
      </div>
    );
  }

  // ── Main workspace ──
  return (
    <div className="flex flex-col flex-1 h-full bg-[#0A0A0A] text-white overflow-hidden">
      {/* Project header */}
      <ProjectHeader
        project={project}
        onDeleteRequest={() => setPendingDelete(true)}
      />

      {/* Scrollable workspace body */}
      <main className="flex-1 overflow-y-auto">
        <div
          style={{
            maxWidth: '1000px',
            width: '100%',
            margin: '0 auto',
            padding: '40px 32px 80px',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {/* New chat composer */}
          <div style={{ marginBottom: '40px' }}>
            <NewChatComposer
              projectName={project.name}
              onSubmit={handleNewChat}
            />
          </div>

          {/* Tabs section */}
          <WorkspaceTabs>
            {(activeTab) =>
              activeTab === 'chats' ? (
                <ChatsTab projectId={project.id} />
              ) : (
                <SourcesTab projectId={project.id} />
              )
            }
          </WorkspaceTabs>
        </div>
      </main>

      {/* Delete workspace confirm */}
      {pendingDelete && (
        <DeleteProjectConfirm
          project={project}
          onConfirm={handleDeleteProject}
          onCancel={() => setPendingDelete(false)}
          deleting={deleting}
        />
      )}
    </div>
  );
};
