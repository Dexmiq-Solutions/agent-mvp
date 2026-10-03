import React, { useState, useCallback } from 'react';
import { Folder, MoreHorizontal, Pencil, Trash2 } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useNavigate } from 'react-router-dom';
import type { Project } from '../../types';

interface ProjectCardProps {
  project: Project;
  onDeleteRequest: (project: Project) => void;
  onRenameRequest: (project: Project) => void;
}

export const ProjectCard: React.FC<ProjectCardProps> = ({
  project,
  onDeleteRequest,
  onRenameRequest,
}) => {
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);
  const [hovered, setHovered] = useState(false);

  const handleCardClick = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('[data-menu]')) return;
    navigate(`/projects/${project.id}`);
  }, [navigate, project.id]);

  const updatedAt = project.updated_at
    ? formatDistanceToNow(new Date(project.updated_at), { addSuffix: true })
    : null;

  return (
    <article
      onClick={handleCardClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      aria-label={`Open project: ${project.name}`}
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleCardClick(e as any); } }}
      role="button"
      className={`
        relative flex flex-col p-6 rounded-2xl cursor-pointer select-none
        transition-all duration-200 ease-out
        bg-gradient-to-b from-zinc-900/80 to-zinc-950/80 backdrop-blur-md
        border border-white/10 border-t-white/15
        shadow-2xl shadow-black/50
        ${hovered ? '-translate-y-1 border-indigo-500/40 bg-gradient-to-b from-zinc-900/90 to-zinc-950/90' : 'translate-y-0'}
      `}
      style={{
        minHeight: '160px',
      }}
    >
      {/* ── Top row: icon + menu trigger ── */}
      <div className="flex items-start justify-between mb-4">

        {/* Project icon badge */}
        <div
          className={`
            w-9 h-9 rounded-lg flex items-center justify-center shrink-0 transition-colors duration-150
            bg-indigo-500/10 text-indigo-400
            ${hovered ? 'bg-indigo-500/20 shadow-[0_0_15px_rgba(99,102,241,0.2)]' : ''}
          `}
        >
          <Folder size={16} strokeWidth={2} />
        </div>

        {/* Action menu */}
        <div data-menu style={{ position: 'relative', flexShrink: 0 }}>
          <button
            onClick={(e) => { e.stopPropagation(); setMenuOpen((v) => !v); }}
            aria-label="Project options"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '26px',
              height: '26px',
              borderRadius: '6px',
              border: menuOpen ? '1px solid #2a2a2a' : '1px solid transparent',
              background: menuOpen ? '#1e1e1e' : 'transparent',
              color: hovered || menuOpen ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0)',
              cursor: 'pointer',
              transition: 'all 0.12s ease',
            }}
          >
            <MoreHorizontal size={15} strokeWidth={2} />
          </button>

          {/* Dropdown */}
          {menuOpen && (
            <>
              {/* Click-outside overlay */}
              <div
                style={{ position: 'fixed', inset: 0, zIndex: 30 }}
                onClick={(e) => { e.stopPropagation(); setMenuOpen(false); }}
              />
              <div
                style={{
                  position: 'absolute',
                  right: 0,
                  top: 'calc(100% + 4px)',
                  minWidth: '152px',
                  background: '#161616',
                  border: '1px solid #2a2a2a',
                  borderRadius: '8px',
                  padding: '3px',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
                  zIndex: 40,
                  animation: 'modal-in 0.15s cubic-bezier(0.16, 1, 0.3, 1) forwards',
                }}
              >
                <DropdownItem
                  icon={<Pencil size={13} strokeWidth={2} />}
                  label="Rename"
                  onClick={() => { onRenameRequest(project); setMenuOpen(false); }}
                />
                <div style={{ height: '1px', background: 'rgba(255,255,255,0.06)', margin: '3px 0' }} />
                <DropdownItem
                  icon={<Trash2 size={13} strokeWidth={2} />}
                  label="Delete"
                  danger
                  onClick={() => { onDeleteRequest(project); setMenuOpen(false); }}
                />
              </div>
            </>
          )}
        </div>
      </div>

      {/* ── Content ── */}
      <div className="flex flex-col flex-1 min-w-0">
        <h3
          style={{
            fontSize: '15px',
            fontWeight: 600,
            color: '#FFFFFF',
            letterSpacing: '-0.02em',
            lineHeight: '1.3',
            marginBottom: '8px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {project.name}
        </h3>

        {project.description ? (
          <p
            style={{
              fontSize: '12.5px',
              color: '#8E8E93',
              lineHeight: '1.5',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
              marginBottom: '0',
              flex: 1,
            }}
          >
            {project.description}
          </p>
        ) : (
          <p
            style={{
              fontSize: '12.5px',
              color: 'rgba(255,255,255,0.2)',
              fontStyle: 'italic',
              flex: 1,
            }}
          >
            No description
          </p>
        )}
      </div>

      <div
        className="pt-4 mt-4 flex items-center justify-between border-t border-white/5"
      >
        <span className="text-[11.5px] text-zinc-500 font-medium tracking-wide">
          {updatedAt ? `Updated ${updatedAt}` : 'Unknown'}
        </span>
        <span
          className={`
            text-[11.5px] font-semibold text-indigo-400 transition-opacity duration-200
            ${hovered ? 'opacity-100' : 'opacity-0'}
          `}
        >
          Open &rarr;
        </span>
      </div>
    </article>
  );
};

/* ── Shared dropdown item ── */
function DropdownItem({
  icon,
  label,
  onClick,
  danger = false,
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
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        width: '100%',
        padding: '6px 9px',
        borderRadius: '6px',
        border: 'none',
        background: hov
          ? danger ? 'rgba(239,68,68,0.1)' : 'rgba(255,255,255,0.05)'
          : 'transparent',
        color: danger
          ? hov ? '#f87171' : 'rgba(252,165,165,0.7)'
          : hov ? '#FFFFFF' : 'rgba(255,255,255,0.6)',
        fontSize: '12.5px',
        fontFamily: 'Inter, sans-serif',
        fontWeight: 500,
        cursor: 'pointer',
        transition: 'background 0.1s ease, color 0.1s ease',
        textAlign: 'left',
      }}
    >
      {icon}
      {label}
    </button>
  );
}
