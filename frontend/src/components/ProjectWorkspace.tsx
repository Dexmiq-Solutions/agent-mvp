import React, { useState, useEffect, useCallback } from 'react';
import type { Project, Conversation, SourceDocument, Message } from '../types';
import { ConversationList } from './ConversationList';
import { SourceList } from './SourceList';
import { ChatView } from './ChatView';
import { listConversations, createConversation, listMessages } from '../api/conversations';
import { listSources, uploadSource, deleteSource } from '../api/sources';

interface ProjectWorkspaceProps {
  project: Project;
  onBackToProjects: () => void;
}

export const ProjectWorkspace: React.FC<ProjectWorkspaceProps> = ({
  project,
  onBackToProjects,
}) => {
  // Conversations state (scoped to project.id)
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loadingConversations, setLoadingConversations] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);

  // Sources state (scoped to project.id)
  const [sources, setSources] = useState<SourceDocument[]>([]);
  const [loadingSources, setLoadingSources] = useState(false);
  const [uploadingSource, setUploadingSource] = useState(false);
  const [sourceError, setSourceError] = useState<string | null>(null);

  // Messages state (scoped to activeConversationId)
  const [messages, setMessages] = useState<Message[]>([]);
  const [loadingMessages, setLoadingMessages] = useState(false);

  // Global workspace error
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);

  // 1. Load conversations for the current project
  const fetchConversations = useCallback(async () => {
    setLoadingConversations(true);
    setWorkspaceError(null);
    try {
      const data = await listConversations(project.id);
      setConversations(data);
      // If there are conversations and none selected, auto-select the first one
      if (data.length > 0 && !activeConversationId) {
        setActiveConversationId(data[0].id);
      }
    } catch (err) {
      setWorkspaceError(err instanceof Error ? err.message : 'Failed to load conversations');
    } finally {
      setLoadingConversations(false);
    }
  }, [project.id, activeConversationId]);

  // 2. Load sources for the current project
  const fetchSources = useCallback(async () => {
    setLoadingSources(true);
    setSourceError(null);
    try {
      const data = await listSources(project.id);
      setSources(data);
    } catch (err) {
      setSourceError(err instanceof Error ? err.message : 'Failed to load sources');
    } finally {
      setLoadingSources(false);
    }
  }, [project.id]);

  // 3. Load messages when active conversation changes
  const fetchMessages = useCallback(async (conversationId: string) => {
    setLoadingMessages(true);
    try {
      const data = await listMessages(project.id, conversationId);
      setMessages(data);
    } catch (err) {
      setWorkspaceError(err instanceof Error ? err.message : 'Failed to load messages');
    } finally {
      setLoadingMessages(false);
    }
  }, [project.id]);

  // Project boundary effect: reset everything when project changes
  useEffect(() => {
    setActiveConversationId(null);
    setMessages([]);
    setConversations([]);
    setSources([]);
    fetchConversations();
    fetchSources();
  }, [project.id]);

  // Conversation selection effect
  useEffect(() => {
    if (activeConversationId) {
      fetchMessages(activeConversationId);
    } else {
      setMessages([]);
    }
  }, [activeConversationId, fetchMessages]);

  // Handlers
  const handleCreateConversation = async (title?: string) => {
    try {
      const newConv = await createConversation(project.id, title);
      setConversations((prev) => [newConv, ...prev]);
      setActiveConversationId(newConv.id);
      setMessages([]);
    } catch (err) {
      setWorkspaceError(err instanceof Error ? err.message : 'Failed to create conversation');
    }
  };

  const handleUploadSource = async (file: File, name?: string) => {
    setUploadingSource(true);
    setSourceError(null);
    try {
      await uploadSource(project.id, file, name);
      // Refresh list to pick up any async processing status
      await fetchSources();
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to upload source';
      setSourceError(msg);
      throw err;
    } finally {
      setUploadingSource(false);
    }
  };

  const handleDeleteSource = async (documentId: string) => {
    setSourceError(null);
    try {
      await deleteSource(project.id, documentId);
      await fetchSources();
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to delete source';
      setSourceError(msg);
      throw err;
    }
  };

  const handleMessageSent = (userMsg: Message, assistantMsg: Message) => {
    setMessages((prev) => {
      // Remove any temp optimistic user message and append authoritative ones
      const filtered = prev.filter((m) => m.id !== userMsg.id);
      return [...filtered, userMsg, assistantMsg];
    });

    // Update conversation message count in list
    setConversations((prev) =>
      prev.map((c) =>
        c.id === activeConversationId
          ? { ...c, messages_count: (c.messages_count ?? 0) + 2 }
          : c
      )
    );
  };

  const activeConversation = conversations.find((c) => c.id === activeConversationId) || null;

  return (
    <div className="workspace-container" id="project-workspace">
      {/* Top Workspace Bar */}
      <header className="workspace-header">
        <div className="workspace-header-left">
          <button
            type="button"
            className="btn btn-secondary btn-back"
            onClick={onBackToProjects}
            id="btn-back-to-projects"
          >
            &larr; Projects
          </button>
          <div className="workspace-title-info">
            <h1 className="project-headline">{project.name}</h1>
            {project.description && (
              <span className="project-subtext">{project.description}</span>
            )}
          </div>
        </div>
        <div className="workspace-header-right">
          <span className="badge badge-boundary" title="Authoritative Project Scope">
            Project Boundary: {project.id.slice(0, 8)}...
          </span>
        </div>
      </header>

      {workspaceError && (
        <div className="alert alert-error workspace-alert" id="workspace-error-banner">
          {workspaceError}
        </div>
      )}

      {/* Main Workspace Body */}
      <div className="workspace-body">
        {/* Left Sidebar: Conversations & Sources */}
        <aside className="workspace-sidebar" id="workspace-sidebar">
          <ConversationList
            conversations={conversations}
            activeConversationId={activeConversationId}
            loading={loadingConversations}
            onSelectConversation={(id) => setActiveConversationId(id)}
            onCreateConversation={handleCreateConversation}
          />

          <SourceList
            sources={sources}
            loading={loadingSources}
            uploading={uploadingSource}
            error={sourceError}
            onUploadSource={handleUploadSource}
            onDeleteSource={handleDeleteSource}
            onRefresh={fetchSources}
          />
        </aside>

        {/* Right Main Pane: Chat with BRD Agent */}
        <main className="workspace-main" id="workspace-main-chat">
          <ChatView
            projectId={project.id}
            conversation={activeConversation}
            messages={messages}
            loadingMessages={loadingMessages}
            onRefreshMessages={() => {
              if (activeConversationId) fetchMessages(activeConversationId);
            }}
            onMessageSent={handleMessageSent}
          />
        </main>
      </div>
    </div>
  );
};
