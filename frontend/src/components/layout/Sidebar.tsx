import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutGrid, Hexagon } from 'lucide-react';

export const Sidebar: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="flex h-screen w-full overflow-hidden" style={{ background: '#0A0A0A' }}>

      {/* ─── Sidebar ─────────────────────────────────────────────────────── */}
      <aside
        className="flex flex-col flex-shrink-0"
        style={{
          width: '220px',
          background: '#111111',
          borderRight: '1px solid #1e1e1e',
        }}
      >
        {/* Brand */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            padding: '20px 16px 20px',
            borderBottom: '1px solid #1a1a1a',
            marginBottom: '8px',
          }}
        >
          <div
            style={{
              width: '26px',
              height: '26px',
              borderRadius: '7px',
              background: 'rgba(99,102,241,0.15)',
              border: '1px solid rgba(99,102,241,0.25)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}
          >
            <Hexagon size={12} strokeWidth={2} color="#a5b4fc" />
          </div>
          <span
            style={{
              fontFamily: 'Inter, sans-serif',
              fontSize: '13.5px',
              fontWeight: 600,
              letterSpacing: '-0.02em',
              color: '#FFFFFF',
            }}
          >
            Dexmiq AI
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
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '7px 10px',
              borderRadius: '7px',
              fontSize: '13px',
              fontWeight: isActive ? 500 : 400,
              letterSpacing: '-0.01em',
              textDecoration: 'none',
              transition: 'background 0.12s ease, color 0.12s ease',
              color: isActive ? '#FFFFFF' : 'rgba(255,255,255,0.45)',
              background: isActive ? 'rgba(255,255,255,0.06)' : 'transparent',
            })}
          >
            {({ isActive }) => (
              <>
                <div
                  style={{
                    width: '3px',
                    height: '14px',
                    borderRadius: '2px',
                    background: isActive ? '#6366f1' : 'transparent',
                    flexShrink: 0,
                    transition: 'background 0.12s ease',
                  }}
                />
                <LayoutGrid
                  size={14}
                  strokeWidth={isActive ? 2 : 1.75}
                  style={{ color: isActive ? '#a5b4fc' : 'rgba(255,255,255,0.4)', flexShrink: 0 }}
                />
                <span>Projects</span>
              </>
            )}
          </NavLink>
        </nav>
      </aside>

      {/* ─── Main content ─────────────────────────────────────────────────── */}
      <main className="flex flex-col flex-1 min-w-0 h-full overflow-hidden">
        {children}
      </main>

    </div>
  );
};
