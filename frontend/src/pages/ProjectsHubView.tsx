import React, { useEffect, useState, useMemo, useRef, useCallback } from 'react';
import { Search, Plus, FolderOpen, AlertCircle, RotateCcw } from 'lucide-react';
import { mockProjectService } from '../services/mockData';
import type { Project } from '../types';
import { ProjectCard } from '../components/projects/ProjectCard';
import { CreateProjectModal } from '../components/projects/CreateProjectModal';
import { DeleteProjectModal } from '../components/projects/DeleteProjectModal';

// ─── Toggle: swap to real services when backend is live ─────────────────────
const api = mockProjectService;

export const ProjectsHubView: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<Project | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  // Fetch on mount
  useEffect(() => { fetchProjects(); }, []);

  const fetchProjects = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.listProjects(100, 0);
      setProjects(res.items);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load projects.');
    } finally {
      setLoading(false);
    }
  };

  const handleRenameRequest = useCallback(async (project: Project) => {
    const newName = window.prompt('Rename project:', project.name);
    if (!newName || newName.trim() === '' || newName.trim() === project.name) return;
    try {
      const updated = await api.updateProject(project.id, { name: newName.trim() });
      setProjects((prev) => prev.map((p) => (p.id === updated.id ? updated : p)));
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Rename failed.');
    }
  }, []);

  const filteredProjects = useMemo(() => {
    if (!searchQuery.trim()) return projects;
    const q = searchQuery.toLowerCase();
    return projects.filter(
      (p) => p.name.toLowerCase().includes(q) || (p.description?.toLowerCase().includes(q) ?? false)
    );
  }, [projects, searchQuery]);

  /* ── Input styling helpers ── */
  const focusInput = (e: React.FocusEvent<HTMLInputElement>) => {
    e.target.style.borderColor = 'rgba(99,102,241,0.5)';
    e.target.style.background = '#1a1a1a';
    e.target.style.boxShadow = '0 0 0 3px rgba(99,102,241,0.1)';
  };
  const blurInput = (e: React.FocusEvent<HTMLInputElement>) => {
    e.target.style.borderColor = '#2a2a2a';
    e.target.style.background = '#141414';
    e.target.style.boxShadow = 'none';
  };

  return (
    <div
      className="flex flex-col flex-1 h-full overflow-y-auto"
      style={{ background: '#0A0A0A' }}
    >

      {/* ─── Sticky page header ──────────────────────────────────────────── */}
      <header
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 10,
          background: 'rgba(10,10,10,0.92)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          borderBottom: '1px solid #1e1e1e',
          padding: '20px 32px',
        }}
      >
        <div
          style={{
            maxWidth: '1200px',
            margin: '0 auto',
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: '24px',
            flexWrap: 'wrap',
          }}
        >
          {/* Title + subtitle */}
          <div>
            <h1
              style={{
                fontSize: '20px',
                fontWeight: 700,
                letterSpacing: '-0.03em',
                color: '#FFFFFF',
                lineHeight: 1.15,
                margin: 0,
                marginBottom: '3px',
              }}
            >
              Projects
            </h1>
            <p
              style={{
                fontSize: '13px',
                color: '#8E8E93',
                letterSpacing: '-0.005em',
                margin: 0,
                lineHeight: 1,
              }}
            >
              Your AI workspaces and knowledge bases
            </p>
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexShrink: 0, flexWrap: 'wrap' }}>

            {/* Search */}
            <div style={{ position: 'relative' }}>
              <Search
                size={14}
                strokeWidth={2}
                style={{
                  position: 'absolute',
                  left: '11px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: 'rgba(148,163,184,0.5)',
                  pointerEvents: 'none',
                }}
              />
              <input
                ref={searchRef}
                type="text"
                placeholder="Search projects..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onFocus={focusInput}
                onBlur={blurInput}
                style={{
                  width: '220px',
                  padding: '7px 44px 7px 30px',
                  borderRadius: '7px',
                  border: '1px solid #2a2a2a',
                  background: '#141414',
                  color: '#FFFFFF',
                  fontSize: '13px',
                  fontFamily: 'Inter, sans-serif',
                  outline: 'none',
                  transition: 'border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease',
                }}
              />
              {/* Keyboard hint */}
              <div
                style={{
                  position: 'absolute',
                  right: '8px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '2px',
                  pointerEvents: 'none',
                }}
              >
                <kbd
                  style={{
                    padding: '2px 5px',
                    borderRadius: '4px',
                    background: 'rgba(255,255,255,0.06)',
                    border: '1px solid rgba(255,255,255,0.09)',
                    fontSize: '10.5px',
                    color: 'rgba(148,163,184,0.5)',
                    fontFamily: 'Inter, sans-serif',
                    lineHeight: '14px',
                    letterSpacing: '0.01em',
                  }}
                >
                  Ctrl K
                </kbd>
              </div>
            </div>

            {/* New Project */}
            <NewProjectButton onClick={() => setIsCreateOpen(true)} />
          </div>
        </div>
      </header>

      {/* ─── Main content ─────────────────────────────────────────────────── */}
      <main
        style={{
          flex: 1,
          maxWidth: '1200px',
          width: '100%',
          margin: '0 auto',
          padding: '28px 32px 64px',
        }}
      >

        {/* Error banner */}
        {error && !loading && (
          <div
            role="alert"
            style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: '12px',
              padding: '14px 16px',
              marginBottom: '24px',
              borderRadius: '8px',
              background: 'rgba(239,68,68,0.06)',
              border: '1px solid rgba(239,68,68,0.15)',
            }}
          >
            <AlertCircle size={14} strokeWidth={2} color="#f87171" style={{ flexShrink: 0, marginTop: '1px' }} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <p style={{ fontSize: '13px', fontWeight: 500, color: '#f87171', marginBottom: '2px' }}>
                Failed to load projects
              </p>
              <p style={{ fontSize: '12.5px', color: 'rgba(252,165,165,0.65)' }}>
                Please check your connection and try again.
              </p>
            </div>
            <button
              onClick={fetchProjects}
              aria-label="Retry loading projects"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '5px',
                padding: '5px 10px',
                borderRadius: '6px',
                border: '1px solid rgba(239,68,68,0.2)',
                background: 'rgba(239,68,68,0.08)',
                color: 'rgba(252,165,165,0.8)',
                fontSize: '12px',
                fontFamily: 'Inter, sans-serif',
                fontWeight: 500,
                cursor: 'pointer',
                flexShrink: 0,
                whiteSpace: 'nowrap',
              }}
            >
              <RotateCcw size={11} strokeWidth={2} />
              Retry
            </button>
          </div>
        )}

        {/* ── Count line ── */}
        {!loading && filteredProjects.length > 0 && (
          <p
            style={{
              fontSize: '11.5px',
              color: 'rgba(255,255,255,0.22)',
              marginBottom: '16px',
              letterSpacing: '0.02em',
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            {filteredProjects.length} {filteredProjects.length === 1 ? 'project' : 'projects'}
            {searchQuery && ` — matching “${searchQuery}”`}
          </p>
        )}

        {/* ── Loading skeletons ── */}
        {loading && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
              gap: '12px',
            }}
          >
            {[...Array(6)].map((_, i) => (
              <div
                key={i}
                className="skeleton"
                style={{
                  height: '148px',
                  padding: '20px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '12px',
                }}
              >
                {/* Icon placeholder */}
                <div
                  style={{
                    width: '36px',
                    height: '36px',
                    borderRadius: '8px',
                    background: 'rgba(255,255,255,0.04)',
                    flexShrink: 0,
                  }}
                />
                {/* Title placeholder */}
                <div
                  style={{
                    height: '12px',
                    width: '55%',
                    borderRadius: '5px',
                    background: 'rgba(255,255,255,0.05)',
                  }}
                />
                {/* Description placeholder */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', flex: 1 }}>
                  <div
                    style={{
                      height: '10px',
                      width: '90%',
                      borderRadius: '4px',
                      background: 'rgba(255,255,255,0.04)',
                    }}
                  />
                  <div
                    style={{
                      height: '10px',
                      width: '70%',
                      borderRadius: '4px',
                      background: 'rgba(255,255,255,0.03)',
                    }}
                  />
                </div>
                {/* Meta placeholder */}
                <div
                  style={{
                    height: '9px',
                    width: '40%',
                    borderRadius: '4px',
                    background: 'rgba(255,255,255,0.03)',
                    marginTop: 'auto',
                  }}
                />
              </div>
            ))}
          </div>
        )}

        {/* ── Project grid ── */}
        {!loading && filteredProjects.length > 0 && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
              gap: '12px',
            }}
          >
            {filteredProjects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                onDeleteRequest={setProjectToDelete}
                onRenameRequest={handleRenameRequest}
              />
            ))}
          </div>
        )}

        {/* ── Empty states ── */}
        {!loading && !error && filteredProjects.length === 0 && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '72px 24px',
              textAlign: 'center',
            }}
          >
            <div
              style={{
                width: '48px',
                height: '48px',
                borderRadius: '12px',
                background: '#141414',
                border: '1px solid #222222',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '18px',
              }}
            >
              <FolderOpen size={20} strokeWidth={1.5} color="rgba(99,102,241,0.6)" />
            </div>

            {searchQuery ? (
              <>
                <h2
                  style={{
                    fontSize: '15px',
                    fontWeight: 600,
                    color: '#FFFFFF',
                    letterSpacing: '-0.02em',
                    marginBottom: '6px',
                  }}
                >
                  No projects found
                </h2>
                <p
                  style={{
                    fontSize: '13px',
                    color: '#8E8E93',
                    maxWidth: '280px',
                    lineHeight: 1.6,
                    marginBottom: '20px',
                  }}
                >
                  No projects match{' '}
                  <span style={{ color: 'rgba(255,255,255,0.6)', fontWeight: 500 }}>“{searchQuery}”</span>.
                </p>
                <button
                  onClick={() => setSearchQuery('')}
                  style={{
                    padding: '6px 14px',
                    borderRadius: '7px',
                    border: '1px solid #2a2a2a',
                    background: '#141414',
                    color: 'rgba(255,255,255,0.55)',
                    fontSize: '13px',
                    fontFamily: 'Inter, sans-serif',
                    fontWeight: 500,
                    cursor: 'pointer',
                    transition: 'background 0.12s ease, border-color 0.12s ease',
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.background = '#1a1a1a';
                    (e.currentTarget as HTMLButtonElement).style.borderColor = '#333';
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.background = '#141414';
                    (e.currentTarget as HTMLButtonElement).style.borderColor = '#2a2a2a';
                  }}
                >
                  Clear search
                </button>
              </>
            ) : (
              <>
                <h2
                  style={{
                    fontSize: '15px',
                    fontWeight: 600,
                    color: '#FFFFFF',
                    letterSpacing: '-0.02em',
                    marginBottom: '6px',
                  }}
                >
                  No projects yet
                </h2>
                <p
                  style={{
                    fontSize: '13px',
                    color: '#8E8E93',
                    maxWidth: '260px',
                    lineHeight: 1.6,
                    marginBottom: '20px',
                  }}
                >
                  Create your first workspace to start organizing your knowledge and chats.
                </p>
                <NewProjectButton onClick={() => setIsCreateOpen(true)} />
              </>
            )}
          </div>
        )}
      </main>

      {/* ─── Modals ── */}
      <CreateProjectModal
        isOpen={isCreateOpen}
        onClose={() => setIsCreateOpen(false)}
        onSuccess={(project) => {
          setProjects((prev) => [project, ...prev]);
          setIsCreateOpen(false);
        }}
      />
      <DeleteProjectModal
        isOpen={!!projectToDelete}
        project={projectToDelete}
        onClose={() => setProjectToDelete(null)}
        onSuccess={(deletedId) => {
          setProjects((prev) => prev.filter((p) => p.id !== deletedId));
          setProjectToDelete(null);
        }}
      />
    </div>
  );
};

/* ── New Project button (shared between header + empty state) ── */
function NewProjectButton({ onClick }: { onClick: () => void }) {
  const [hov, setHov] = useState(false);
  const [pressed, setPressed] = useState(false);

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => { setHov(false); setPressed(false); }}
      onMouseDown={() => setPressed(true)}
      onMouseUp={() => setPressed(false)}
      aria-label="Create a new project"
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        padding: '7px 13px',
        borderRadius: '7px',
        border: '1px solid rgba(99,102,241,0.35)',
        background: hov ? 'rgba(99,102,241,0.2)' : 'rgba(99,102,241,0.12)',
        color: hov ? '#c7d2fe' : '#a5b4fc',
        fontSize: '13px',
        fontFamily: 'Inter, sans-serif',
        fontWeight: 500,
        letterSpacing: '-0.01em',
        cursor: 'pointer',
        transition: 'background 0.12s ease, color 0.12s ease, transform 0.1s ease',
        transform: pressed ? 'scale(0.97)' : 'scale(1)',
        whiteSpace: 'nowrap',
      }}
    >
      <Plus size={13} strokeWidth={2.5} />
      New project
    </button>
  );
}
