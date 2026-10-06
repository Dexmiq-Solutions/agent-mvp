import React, { useState } from 'react';
import type { Project, ProjectCreatePayload } from '../types';

interface ProjectListProps {
  projects: Project[];
  loading: boolean;
  error: string | null;
  onSelectProject: (project: Project) => void;
  onCreateProject: (payload: ProjectCreatePayload) => Promise<void>;
  onRefresh: () => void;
}

export const ProjectList: React.FC<ProjectListProps> = ({
  projects,
  loading,
  error,
  onSelectProject,
  onCreateProject,
  onRefresh,
}) => {
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setFormError('Project name is required');
      return;
    }

    setSubmitting(true);
    setFormError(null);
    try {
      await onCreateProject({
        name: name.trim(),
        description: description.trim() || undefined,
      });
      setName('');
      setDescription('');
      setShowCreateForm(false);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Failed to create project');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="project-list-container" id="project-list-view">
      <div className="section-header">
        <div>
          <h2>Projects</h2>
          <p className="subtitle">Select an existing project or create a new one to begin.</p>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={onRefresh}
            disabled={loading}
            id="btn-refresh-projects"
          >
            Refresh
          </button>
          {!showCreateForm && (
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => setShowCreateForm(true)}
              id="btn-open-create-project"
            >
              + New Project
            </button>
          )}
        </div>
      </div>

      {error && <div className="alert alert-error" id="projects-error-banner">{error}</div>}

      {showCreateForm && (
        <div className="card create-form-card" id="create-project-card">
          <h3>Create New Project</h3>
          {formError && <div className="alert alert-error">{formError}</div>}
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="project-name-input">Project Name *</label>
              <input
                id="project-name-input"
                type="text"
                placeholder="e.g. Healthcare Claims Automation"
                value={name}
                onChange={(e) => setName(e.target.value)}
                disabled={submitting}
                autoFocus
              />
            </div>
            <div className="form-group">
              <label htmlFor="project-desc-input">Description (Optional)</label>
              <textarea
                id="project-desc-input"
                rows={2}
                placeholder="Brief summary of this project scope..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                disabled={submitting}
              />
            </div>
            <div className="form-actions">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  setShowCreateForm(false);
                  setFormError(null);
                }}
                disabled={submitting}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={submitting || !name.trim()}
                id="btn-submit-create-project"
              >
                {submitting ? 'Creating...' : 'Create Project'}
              </button>
            </div>
          </form>
        </div>
      )}

      {loading && projects.length === 0 ? (
        <div className="loading-state" id="projects-loading-indicator">
          <div className="spinner"></div>
          <span>Loading projects...</span>
        </div>
      ) : projects.length === 0 ? (
        <div className="empty-state" id="projects-empty-state">
          <p>No projects found.</p>
          {!showCreateForm && (
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => setShowCreateForm(true)}
            >
              + Create Your First Project
            </button>
          )}
        </div>
      ) : (
        <div className="project-grid" id="project-items-grid">
          {projects.map((project) => (
            <div
              key={project.id}
              className="card project-card"
              onClick={() => onSelectProject(project)}
              id={`project-card-${project.id}`}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  onSelectProject(project);
                }
              }}
            >
              <div className="project-card-header">
                <h3 className="project-title">{project.name}</h3>
                <span className="badge badge-id" title={project.id}>
                  ID: {project.id.slice(0, 8)}...
                </span>
              </div>
              {project.description && (
                <p className="project-description">{project.description}</p>
              )}
              <div className="project-card-footer">
                <span className="project-meta">
                  {project.created_at
                    ? `Created: ${new Date(project.created_at).toLocaleDateString()}`
                    : ''}
                </span>
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectProject(project);
                  }}
                  id={`btn-open-project-${project.id}`}
                >
                  Open Workspace &rarr;
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
