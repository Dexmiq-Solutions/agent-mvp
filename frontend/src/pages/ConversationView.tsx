import React from 'react';
import { useParams, useLocation, Link } from 'react-router-dom';
import { ArrowLeft, MessageSquare } from 'lucide-react';

/**
 * ConversationView — STEP 3 placeholder.
 *
 * Receives initialPrompt from ProjectHomeView navigation state.
 * The actual chat/message generation logic will be implemented in STEP 3.
 */
export const ConversationView: React.FC = () => {
  const { projectId, conversationId } = useParams<{ projectId: string; conversationId: string }>();
  const location = useLocation();
  const initialPrompt = (location.state as { initialPrompt?: string } | null)?.initialPrompt;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        height: '100%',
        background: '#0A0A0A',
        overflow: 'hidden',
      }}
    >
      {/* Simple header */}
      <header
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          height: '52px',
          padding: '0 20px',
          borderBottom: '1px solid #1e1e1e',
          flexShrink: 0,
        }}
      >
        <Link
          to={`/projects/${projectId}`}
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: '28px', height: '28px', borderRadius: '7px',
            border: '1px solid #222222', background: '#141414',
            color: '#8E8E93', textDecoration: 'none',
            transition: 'all 0.12s ease',
          }}
          aria-label="Back to workspace"
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLAnchorElement).style.color = '#FFFFFF';
            (e.currentTarget as HTMLAnchorElement).style.borderColor = '#333333';
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLAnchorElement).style.color = '#8E8E93';
            (e.currentTarget as HTMLAnchorElement).style.borderColor = '#222222';
          }}
        >
          <ArrowLeft size={13} strokeWidth={2} />
        </Link>
        <div
          style={{
            width: '26px', height: '26px', borderRadius: '7px',
            background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.2)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
          }}
        >
          <MessageSquare size={13} strokeWidth={1.75} color="#a5b4fc" />
        </div>
        <p style={{ fontSize: '13.5px', fontWeight: 500, color: '#FFFFFF', letterSpacing: '-0.01em' }}>
          Conversation
        </p>
        <p style={{ fontSize: '11px', color: '#8E8E93' }}>
          {conversationId?.slice(0, 8)}…
        </p>
      </header>

      {/* Placeholder body */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '32px 24px',
          textAlign: 'center',
          gap: '12px',
        }}
      >
        <div
          style={{
            width: '48px', height: '48px', borderRadius: '12px',
            background: '#141414', border: '1px solid #222222',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            marginBottom: '4px',
          }}
        >
          <MessageSquare size={20} strokeWidth={1.5} color="#8E8E93" />
        </div>
        <p style={{ fontSize: '16px', fontWeight: 600, color: '#FFFFFF', letterSpacing: '-0.02em' }}>
          Conversation View
        </p>
        <p style={{ fontSize: '13px', color: '#8E8E93', maxWidth: '300px', lineHeight: 1.6 }}>
          This is a Step 3 placeholder. The full conversation UI — messages, AI response generation, and RAG sources — will be implemented in the next step.
        </p>
        {initialPrompt && (
          <div
            style={{
              marginTop: '8px',
              padding: '12px 16px',
              background: '#141414',
              border: '1px solid #222222',
              borderRadius: '10px',
              maxWidth: '480px',
              textAlign: 'left',
            }}
          >
            <p style={{ fontSize: '11px', color: '#8E8E93', marginBottom: '5px', fontWeight: 500 }}>Initial prompt received:</p>
            <p style={{ fontSize: '13px', color: '#FFFFFF', lineHeight: 1.55, fontStyle: 'italic' }}>
              "{initialPrompt}"
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
