import React, { useState } from 'react';
import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import { LayoutGrid, Hexagon, Menu, X, Folder, LogOut } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import * as projectService from '../../services/projectService';
import { listConversations } from '../../services/chatService';
import type { Project, Conversation } from '../../types';

export const Sidebar: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [currentProject, setCurrentProject] = useState<Project | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoadingConversations, setIsLoadingConversations] = useState(false);

  const handleLogout = async () => {
    await logout();
    navigate('/login', { replace: true });
  };


  React.useEffect(() => {
    const match = location.pathname.match(/^\/projects\/([a-zA-Z0-9-]+)/);
    if (match) {
      const projectId = match[1];
      // Only fetch if it's a new project ID or we don't have one
      if (currentProject?.id !== projectId) {
        projectService.getProject(projectId).then(setCurrentProject).catch(() => setCurrentProject(null));
      }
    } else {
      setCurrentProject(null);
    }
  }, [location.pathname, currentProject?.id]);

  React.useEffect(() => {
    if (currentProject?.id) {
      setIsLoadingConversations(true);
      listConversations(currentProject.id)
        .then(res => setConversations(res.items || []))
        .catch(console.error)
        .finally(() => setIsLoadingConversations(false));
    } else {
      setConversations([]);
    }
  }, [currentProject?.id, location.pathname]);

  return (
    <div className="flex h-full w-full overflow-hidden" style={{ background: '#0A0A0A' }}>

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
          width: '260px',
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
          <div className="fade-in flex flex-col flex-1 overflow-hidden" style={{ animation: 'fade-in 0.2s ease forwards', minHeight: 0 }}>
            <div style={{ height: '1px', background: '#1a1a1a', margin: '16px' }} />

            {/* Section label */}
            <div style={{ padding: '0 16px 8px' }}>
              <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.08em', color: '#666', textTransform: 'uppercase' }}>
                Current Project
              </span>
            </div>

            <nav
              style={{
                display: 'flex',
                flexDirection: 'column',
                flex: 1,
                overflow: 'hidden',
                padding: '0 8px',
                minHeight: 0,
              }}
            >
              {/* Project Title */}
              <NavLink
                to={`/projects/${currentProject.id}`}
                onClick={() => setMobileOpen(false)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  padding: '10px 12px',
                  borderRadius: '8px',
                  fontSize: '14px',
                  fontWeight: 600,
                  color: '#FFFFFF',
                  marginBottom: '12px',
                  textDecoration: 'none',
                  transition: 'background 0.15s ease',
                }}
                className="hover:bg-white/5"
              >
                <Folder
                  size={18}
                  strokeWidth={2}
                  style={{ color: '#a5b4fc', flexShrink: 0 }}
                />
                <span
                  style={{
                    whiteSpace: 'normal',
                    lineHeight: '1.4',
                    wordBreak: 'break-word',
                  }}
                >
                  {currentProject.name}
                </span>
              </NavLink>

              {/* Chats heading */}
              <div style={{ padding: '0 12px 6px' }}>
                <span style={{ fontSize: '11px', fontWeight: 600, letterSpacing: '0.06em', color: '#666', textTransform: 'uppercase' }}>
                  Chats
                </span>
              </div>

              {/* Conversation List */}
              <div className="flex-1 overflow-y-auto scroll-smooth" style={{ display: 'flex', flexDirection: 'column', gap: '4px', paddingBottom: '16px' }}>
                {isLoadingConversations ? (
                  <span style={{ fontSize: '13px', color: 'rgba(255,255,255,0.4)', padding: '8px 12px' }}>Loading chats...</span>
                ) : conversations.length === 0 ? (
                  <span style={{ fontSize: '13px', color: 'rgba(255,255,255,0.4)', padding: '8px 12px' }}>No chats yet</span>
                ) : (
                  conversations.map(conv => {
                    const isActive = location.pathname === `/projects/${currentProject.id}/c/${conv.id}`;
                    return (
                      <NavLink
                        key={conv.id}
                        to={`/projects/${currentProject.id}/c/${conv.id}`}
                        onClick={() => setMobileOpen(false)}
                        className={`group truncate shrink-0 transition-all duration-200 ${isActive ? 'bg-white/10 text-white shadow-sm' : 'hover:bg-white/5 text-zinc-400 hover:text-zinc-200'}`}
                        title={conv.title || 'New Conversation'}
                        style={{
                          display: 'block',
                          padding: '8px 12px',
                          borderRadius: '8px',
                          fontSize: '13px',
                          fontWeight: isActive ? 500 : 400,
                          textDecoration: 'none',
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          lineHeight: '1.4',
                          minWidth: 0
                        }}
                      >
                        {conv.title || 'New Conversation'}
                      </NavLink>
                    );
                  })
                )}
              </div>
            </nav>
          </div>
        )}

        {/* Bottom spacer — future nav items can go here */}
        <div style={{ marginTop: 'auto' }} />

        {/* User Profile & Sign Out */}
        {user && (
          <div style={{ padding: '8px 12px', borderTop: '1px solid #1a1a1a', flexShrink: 0 }}>
            <div className="flex items-center justify-between gap-2 p-2 rounded-lg bg-white/[0.02] border border-white/[0.04]">
              <div className="flex items-center gap-2.5 min-w-0">
                <div className="w-6 h-6 rounded-md bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center shrink-0 text-indigo-300 text-[11px] font-semibold">
                  {user.email.charAt(0).toUpperCase()}
                </div>
                <span className="text-[12px] text-zinc-300 truncate" title={user.email}>
                  {user.email}
                </span>
              </div>
              <button
                onClick={handleLogout}
                title="Sign out"
                aria-label="Sign out"
                className="p-1 rounded-md text-zinc-400 hover:text-red-400 hover:bg-red-500/10 transition-colors shrink-0"
              >
                <LogOut size={14} />
              </button>
            </div>
          </div>
        )}

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
