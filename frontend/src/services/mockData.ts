// =============================================================================
// Mock Data — Standalone frontend development without backend
// =============================================================================

import type { Project, PaginatedResponse, SourceDocument, Conversation } from '../types';

const MOCK_PROJECTS: Project[] = [
  {
    id: '1a2b3c4d-0000-0000-0000-000000000001',
    name: 'Q4 Marketing Campaign',
    description: 'All assets, briefs, and research for the Q4 2026 campaign launch.',
    created_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '1a2b3c4d-0000-0000-0000-000000000002',
    name: 'Product Roadmap 2027',
    description: 'Feature planning, spec documents, and stakeholder alignment for next year.',
    created_at: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '1a2b3c4d-0000-0000-0000-000000000003',
    name: 'Legal & Compliance Review',
    description: null,
    created_at: new Date(Date.now() - 20 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '1a2b3c4d-0000-0000-0000-000000000004',
    name: 'Onboarding Knowledge Base',
    description: 'Internal documentation, SOPs, and onboarding materials for new hires.',
    created_at: new Date(Date.now() - 35 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '1a2b3c4d-0000-0000-0000-000000000005',
    name: 'Competitor Analysis',
    description: 'Market research and competitive landscape documents.',
    created_at: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000).toISOString(),
  },
];

// In-memory store so create/delete/rename work during the session
let mockProjects: Project[] = [...MOCK_PROJECTS];

function delay(ms = 400) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export const mockProjectService = {
  async listProjects(limit = 100, offset = 0): Promise<PaginatedResponse<Project>> {
    await delay();
    const items = mockProjects.slice(offset, offset + limit);
    return { items, total: mockProjects.length, limit, offset };
  },

  async getProject(id: string): Promise<Project> {
    await delay(200);
    const project = mockProjects.find((p) => p.id === id);
    if (!project) throw new Error(`Project ${id} not found`);
    return project;
  },

  async createProject(payload: { name: string; description?: string | null }): Promise<Project> {
    await delay(600);
    const now = new Date().toISOString();
    const project: Project = {
      id: crypto.randomUUID(),
      name: payload.name,
      description: payload.description ?? null,
      created_at: now,
      updated_at: now,
    };
    mockProjects = [project, ...mockProjects];
    return project;
  },

  async updateProject(id: string, payload: { name?: string; description?: string | null }): Promise<Project> {
    await delay(400);
    mockProjects = mockProjects.map((p) =>
      p.id === id ? { ...p, ...payload, updated_at: new Date().toISOString() } : p
    );
    const updated = mockProjects.find((p) => p.id === id);
    if (!updated) throw new Error(`Project ${id} not found`);
    return updated;
  },

  async deleteProject(id: string): Promise<void> {
    await delay(500);
    mockProjects = mockProjects.filter((p) => p.id !== id);
  },
};

let mockConversations: Conversation[] = [];

export const mockChatService = {
  async listConversations(projectId: string, limit = 20, offset = 0) {
    await delay(300);
    const items = mockConversations
      .filter((c) => c.project_id === projectId)
      .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
      .slice(offset, offset + limit);
    return { items, total: mockConversations.filter(c => c.project_id === projectId).length, limit, offset };
  },

  async getConversation(projectId: string, conversationId: string) {
    await delay(200);
    const conv = mockConversations.find(c => c.id === conversationId && c.project_id === projectId);
    if (!conv) throw new Error('404 Not Found');
    return conv;
  },

  async createConversation(projectId: string, payload: { title?: string }) {
    await delay(400);
    const now = new Date().toISOString();
    const conv: Conversation = {
      id: crypto.randomUUID(),
      project_id: projectId,
      title: payload.title || 'New Conversation',
      created_at: now,
      updated_at: now,
      messages_count: 0,
      messages: []
    };
    mockConversations = [conv, ...mockConversations];
    return conv;
  },

  async updateConversation(projectId: string, conversationId: string, payload: { title: string }) {
    await delay(300);
    mockConversations = mockConversations.map(c => 
      (c.id === conversationId && c.project_id === projectId) 
        ? { ...c, title: payload.title, updated_at: new Date().toISOString() } 
        : c
    );
    const updated = mockConversations.find(c => c.id === conversationId);
    if (!updated) throw new Error('404 Not Found');
    return updated;
  },

  async deleteConversation(projectId: string, conversationId: string) {
    await delay(400);
    mockConversations = mockConversations.filter(c => !(c.id === conversationId && c.project_id === projectId));
  },

  async sendMessage(projectId: string, conversationId: string, payload: { content: string }, generate: boolean = true) {
    // Simulate realistic latency for LLM calls
    await delay(1200);
    const conv = mockConversations.find(c => c.id === conversationId && c.project_id === projectId);
    if (!conv) throw new Error('404 Not Found');
    
    // Simulate adding user message silently to our store
    const now = new Date().toISOString();
    const userMsg = {
      id: crypto.randomUUID(),
      conversation_id: conversationId,
      role: 'user' as const,
      content: payload.content,
      created_at: now,
      metadata: {}
    };
    conv.messages.push(userMsg);
    conv.messages_count++;
    
    if (generate) {

      // Build mock markdown response
      const lines = [
        `Here is a **mock response** to your question about: *"${payload.content}"*`,
        '',
        'I am running **locally without a backend**. When connected to the real API:',
        '',
        '- The AI will search through your uploaded project sources',
        '- Responses will be grounded in retrieved document chunks',
        '- You will see live telemetry data below each message',
        '',
        '```python',
        '# Example code block with copy button',
        'def greet(name: str) -> str:',
        '    return f"Hello, {name}!"',
        '',
        'print(greet("Dexmiq"))',
        '```',
        '',
        'You can also use `inline code` and **bold** or *italic* text.',
      ];

      const aiMsg = {
        id: crypto.randomUUID(),
        conversation_id: conversationId,
        role: 'assistant' as const,
        content: lines.join('\n'),
        created_at: new Date().toISOString(),
        metadata: {},
      };
      conv.messages.push(aiMsg);
      conv.messages_count++;
      conv.updated_at = new Date().toISOString();
      return aiMsg;
    }
    
    conv.updated_at = new Date().toISOString();
    return userMsg;
  }
};

let mockSources: SourceDocument[] = [];

export const mockSourceService = {
  async listSources(projectId: string, limit = 50, offset = 0) {
    await delay(300);
    const items = mockSources.filter(s => s.project_id === projectId).slice(offset, offset + limit);
    return { items, total: mockSources.filter(s => s.project_id === projectId).length, limit, offset };
  },

  async uploadSource(projectId: string, file: File) {
    await delay(800);
    const now = new Date().toISOString();
    const version = {
      id: crypto.randomUUID(),
      document_id: crypto.randomUUID(),
      project_id: projectId,
      version_number: 1,
      original_filename: file.name,
      content_type: file.type || 'text/plain',
      size_bytes: file.size,
      status: 'indexing' as const,
      error_message: null,
      indexed_at: null,
      created_at: now,
      updated_at: now,
    };
    const source: SourceDocument = {
      id: version.document_id,
      project_id: projectId,
      name: file.name,
      created_at: now,
      updated_at: now,
      latest_version: version,
      versions_count: 1,
    };
    mockSources = [source, ...mockSources];
    
    setTimeout(() => {
      const s = mockSources.find(doc => doc.id === source.id);
      if (s && s.latest_version) {
        s.latest_version.status = 'ready';
        s.latest_version.indexed_at = new Date().toISOString();
      }
    }, 4000);
    
    return source;
  },

  async updateSource(_projectId: string, sourceId: string, payload: { name: string }) {
    await delay(300);
    const s = mockSources.find(doc => doc.id === sourceId);
    if (!s) throw new Error('Source not found');
    s.name = payload.name;
    s.updated_at = new Date().toISOString();
    return { ...s };
  },

  async deleteSource(_projectId: string, sourceId: string) {
    await delay(400);
    mockSources = mockSources.filter(s => s.id !== sourceId);
  },

  async uploadSourceVersion(projectId: string, sourceId: string, file: File) {
    await delay(800);
    const s = mockSources.find(doc => doc.id === sourceId);
    if (!s) throw new Error('Source not found');
    const now = new Date().toISOString();
    const version = {
      id: crypto.randomUUID(),
      document_id: sourceId,
      project_id: projectId,
      version_number: s.versions_count + 1,
      original_filename: file.name,
      content_type: file.type || 'text/plain',
      size_bytes: file.size,
      status: 'indexing' as const,
      error_message: null,
      indexed_at: null,
      created_at: now,
      updated_at: now,
    };
    s.latest_version = version;
    s.versions_count += 1;
    s.updated_at = now;
    
    setTimeout(() => {
      const src = mockSources.find(doc => doc.id === sourceId);
      if (src && src.latest_version && src.latest_version.id === version.id) {
        src.latest_version.status = 'ready';
        src.latest_version.indexed_at = new Date().toISOString();
      }
    }, 4000);
    
    return version;
  }
};
