import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { LayoutGrid, Hexagon, Menu, X, Folder } from 'lucide-react';
import { mockProjectService } from '../../services/mockData';
import type { Project } from '../../types';

export const Sidebar: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  const [currentProject, setCurrentProject] = useState<Project | null>(null);

  React.useEffect(() => {
    const match = location.pathname.match(/^\/projects\/([a-zA-Z0-9-]+)/);
    if (match) {
      const projectId = match[1];
      // Only fetch if it's a new project ID or we don't have one
      if (currentProject?.id !== projectId) {
        mockProjectService.getProject(projectId).then(setCurrentProject).catch(() => setCurrentProject(null));
      }
    } else {
      setCurrentProject(null);
    }
  }, [location.pathname, currentProject?.id]);

  return (
    <div className="flex h-screen w-full overflow-hidden" style={{ background: '#0A0A0A' }}>

      {/* ─── Mobile hamburger ──────────────────────────────────────────────── */}
      <button
        onClick={() => setMobileOpen(true)}
        aria-label="Open navigation"
        className="md:hidden fixed top-3.5 left-3.5 z-50 flex items-center justify-center w-8 h-8 rounded-lg bg-[#141414] border border-[#222] text-[#8E8E93] hover:text-white transition-colors"
      >
        <Menu size={15} strokeWidth={2} />
      </button>

      {/* ─── Mobile overlay ────────────────────────────────────────────────── */}
      {mobileOpen && (
        <div
          className="backdrop-enter fixed inset-0 z-40 bg-black/70 md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* ─── Sidebar ──────────────────────────────────────────────────────── */}
      <aside
        className={`
          fixed md:static inset-y-0 left-0 z-50
          flex flex-col flex-shrink-0 h-full
          transition-transform duration-200 ease-out
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full'}
          md:translate-x-0
          bg-zinc-950/50 backdrop-blur-xl border-r border-white/5
        `}
        style={{
          width: '212px',
        }}
        aria-label="Sidebar navigation"
      >
        {/* Brand */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '18px 16px',
            borderBottom: '1px solid #1a1a1a',
            marginBottom: '6px',
            flexShrink: 0,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '9px' }}>
            <div
              className="w-[26px] h-[26px] rounded-lg flex items-center justify-center shrink-0
                         bg-indigo-500/20 border border-indigo-500/30 shadow-[0_0_12px_rgba(99,102,241,0.4)]"
            >
              <Hexagon size={14} strokeWidth={2.5} className="text-indigo-400" />
            </div>
            <span
              style={{
                fontFamily: 'Inter, sans-serif',
                fontSize: '14.5px',
                fontWeight: 700,
                letterSpacing: '-0.02em',
                color: '#FFFFFF',
              }}
            >
              Dexmiq AI
            </span>
          </div>

          {/* Close button — mobile only */}
          <button
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
            className="md:hidden flex items-center justify-center w-7 h-7 rounded-md text-[#8E8E93] hover:text-white hover:bg-white/5 transition-colors"
          >
            <X size={14} strokeWidth={2} />
          </button>
        </div>

        {/* Section label */}
        <div style={{ padding: '0 16px 6px' }}>
          <span style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.08em', color: '#3a3a3a', textTransform: 'uppercase' }}>
            Navigation
          </span>
        </div>

        {/* Nav items */}
        <nav
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '2px',
            padding: '0 8px',
          }}
        >
          <NavLink
            to="/projects"
            onClick={() => setMobileOpen(false)}
            className={({ isActive }) => `
              relative flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-[13px] tracking-tight transition-all duration-200
              ${isActive ? 'font-semibold text-white bg-gradient-to-r from-indigo-500/20 to-transparent' : 'font-medium text-zinc-400 hover:text-zinc-200 hover:bg-white/5'}
            `}
          >
            {({ isActive }) => (
              <>
                <div
                  className={`
                    absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-4 rounded-r-md transition-all duration-200
                    ${isActive ? 'bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.6)]' : 'bg-transparent'}
                  `}
                />
                <LayoutGrid
                  size={16}
                  strokeWidth={isActive ? 2 : 1.75}
                  className={`shrink-0 transition-colors duration-200 ${isActive ? 'text-indigo-400' : 'text-zinc-500'}`}
                />
                <span>Projects</span>
              </>
            )}
          </NavLink>
        </nav>

        {/* Current Project Context */}
        {currentProject && (
          <div className="fade-in" style={{ animation: 'fade-in 0.2s ease forwards' }}>
            <div style={{ height: '1px', background: '#1a1a1a', margin: '16px' }} />

            {/* Section label */}
            <div style={{ padding: '0 16px 6px' }}>
              <span style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.08em', color: '#3a3a3a', textTransform: 'uppercase' }}>
                Current Project
              </span>
            </div>

            <nav
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '2px',
                padding: '0 8px',
              }}
            >
              {/* Project Title */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '7px 10px',
                  borderRadius: '7px',
                  fontSize: '13px',
                  fontWeight: 600,
                  letterSpacing: '-0.01em',
                  color: '#FFFFFF',
                }}
              >
                <Folder
                  size={16}
                  strokeWidth={2}
                  style={{ color: '#a5b4fc', flexShrink: 0 }}
                />
                <span
                  style={{
                    whiteSpace: 'normal',
                    lineHeight: '1.3',
                    wordBreak: 'break-word',
                  }}
                >
                  {currentProject.name}
                </span>
              </div>

              {/* Chats Link */}
              <div style={{ paddingLeft: '32px', marginTop: '2px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                <NavLink
                  to={`/projects/${currentProject.id}?tab=chats`}
                  onClick={() => setMobileOpen(false)}
                  className="hover:bg-white/5 hover:text-white/80 group"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '6px 8px',
                    borderRadius: '6px',
                    fontSize: '12.5px',
                    fontWeight: (location.pathname === `/projects/${currentProject.id}` && (!location.search || location.search.includes('tab=chats'))) || location.pathname.includes('/c/') ? 600 : 400,
                    textDecoration: 'none',
                    transition: 'background 0.12s ease, color 0.12s ease',
                    color: (location.pathname === `/projects/${currentProject.id}` && (!location.search || location.search.includes('tab=chats'))) || location.pathname.includes('/c/') ? '#FFFFFF' : 'rgba(255,255,255,0.45)',
                    background: (location.pathname === `/projects/${currentProject.id}` && (!location.search || location.search.includes('tab=chats'))) || location.pathname.includes('/c/') ? 'rgba(255,255,255,0.07)' : 'transparent',
                  }}
                >
                  <span>Chats</span>
                </NavLink>

                {/* Sources Link */}
                <NavLink
                  to={`/projects/${currentProject.id}?tab=sources`}
                  onClick={() => setMobileOpen(false)}
                  className="hover:bg-white/5 hover:text-white/80 group"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '6px 8px',
                    borderRadius: '6px',
                    fontSize: '12.5px',
                    fontWeight: location.pathname === `/projects/${currentProject.id}` && location.search.includes('tab=sources') ? 600 : 400,
                    textDecoration: 'none',
                    transition: 'background 0.12s ease, color 0.12s ease',
                    color: location.pathname === `/projects/${currentProject.id}` && location.search.includes('tab=sources') ? '#FFFFFF' : 'rgba(255,255,255,0.45)',
                    background: location.pathname === `/projects/${currentProject.id}` && location.search.includes('tab=sources') ? 'rgba(255,255,255,0.07)' : 'transparent',
                  }}
                >
                  <span>Sources</span>
                </NavLink>
              </div>
            </nav>
          </div>
        )}

        {/* Bottom spacer — future nav items can go here */}
        <div style={{ flex: 1 }} />

        {/* Version badge */}
        <div style={{ padding: '12px 16px', borderTop: '1px solid #1a1a1a', flexShrink: 0 }}>
          <span style={{ fontSize: '10.5px', color: '#333', letterSpacing: '0.01em' }}>
            Dexmiq AI · Preview
          </span>
        </div>
      </aside>

      {/* ─── Main content ─────────────────────────────────────────────────── */}
      <main className="flex flex-col flex-1 min-w-0 h-full overflow-hidden md:ml-0">
        {children}
      </main>

    </div>
  );
};
