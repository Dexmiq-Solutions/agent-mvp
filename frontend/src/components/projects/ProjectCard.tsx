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
      style={{
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        padding: '18px',
        borderRadius: '10px',
        border: hovered
          ? '1px solid #2e2e2e'
          : '1px solid #1e1e1e',
        background: hovered ? '#191919' : '#141414',
        cursor: 'pointer',
        transition: 'background 0.15s ease, border-color 0.15s ease',
        minHeight: '148px',
        userSelect: 'none',
      }}
    >
      {/* ── Top row: icon + menu trigger ── */}
      <div className="flex items-start justify-between mb-4">

        {/* Project icon badge */}
        <div
          style={{
            width: '36px',
            height: '36px',
            borderRadius: '8px',
            background: hovered ? 'rgba(99,102,241,0.12)' : 'rgba(99,102,241,0.07)',
            border: hovered ? '1px solid rgba(99,102,241,0.25)' : '1px solid rgba(99,102,241,0.14)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            transition: 'background 0.15s ease, border-color 0.15s ease',
          }}
        >
          <Folder
            size={15}
            strokeWidth={1.75}
            style={{ color: hovered ? 'rgba(165,180,252,0.95)' : 'rgba(165,180,252,0.65)' }}
          />
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
            fontSize: '13.5px',
            fontWeight: 600,
            color: '#FFFFFF',
            letterSpacing: '-0.015em',
            lineHeight: '1.25',
            marginBottom: '6px',
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

      {/* ── Footer metadata ── */}
      <div
        style={{
          paddingTop: '12px',
          marginTop: '12px',
          borderTop: '1px solid #1e1e1e',
        }}
      >
        <span
          style={{
            fontSize: '11px',
            color: 'rgba(255,255,255,0.25)',
            letterSpacing: '0.01em',
          }}
        >
          {updatedAt ? `Updated ${updatedAt}` : 'Unknown'}
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
