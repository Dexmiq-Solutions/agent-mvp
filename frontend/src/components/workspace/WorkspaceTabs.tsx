import React from 'react';
import { useSearchParams } from 'react-router-dom';

type Tab = 'chats' | 'sources';

interface WorkspaceTabsProps {
  children: (activeTab: Tab) => React.ReactNode;
}

export const WorkspaceTabs: React.FC<WorkspaceTabsProps> = ({ children }) => {
  const [searchParams, setSearchParams] = useSearchParams();
  const rawTab = searchParams.get('tab');
  const activeTab: Tab = rawTab === 'sources' ? 'sources' : 'chats';

  const handleTabChange = (tab: Tab) => {
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.set('tab', tab);
        return next;
      },
      { replace: true }
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      {/* Tab bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '16px',
          borderBottom: '1px solid #222222',
          flexShrink: 0,
        }}
      >
        {(['chats', 'sources'] as Tab[]).map((tab) => {
          const isActive = tab === activeTab;
          return (
            <button
              key={tab}
              onClick={() => handleTabChange(tab)}
              style={{
                padding: '16px 0',
                marginRight: '8px',
                fontSize: '14px',
                fontWeight: isActive ? 600 : 500,
                fontFamily: 'Inter, sans-serif',
                letterSpacing: '-0.01em',
                color: isActive ? '#FFFFFF' : '#8E8E93',
                background: 'transparent',
                border: 'none',
                borderBottom: isActive ? '2px solid #6366f1' : '2px solid transparent',
                marginBottom: '-1px',
                cursor: 'pointer',
                transition: 'color 0.15s ease, border-color 0.15s ease',
                textTransform: 'capitalize',
              }}
              onMouseEnter={(e) => {
                if (!isActive) (e.currentTarget as HTMLButtonElement).style.color = '#FFFFFF';
              }}
              onMouseLeave={(e) => {
                if (!isActive) (e.currentTarget as HTMLButtonElement).style.color = '#8E8E93';
              }}
            >
              {tab === 'chats' ? 'Chats' : 'Sources'}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
        {children(activeTab)}
      </div>
    </div>
  );
};
