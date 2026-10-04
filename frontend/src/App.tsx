import React, { useState, useEffect, useCallback } from 'react';
import { Project, ProjectCreatePayload } from './types';
import { listProjects, createProject } from './api/projects';
import { ProjectList } from './components/ProjectList';
import { ProjectWorkspace } from './components/ProjectWorkspace';

export const App: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchProjects = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listProjects();
      setProjects(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to connect to backend server. Make sure the FastAPI backend is running.'
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  const handleCreateProject = async (payload: ProjectCreatePayload) => {
    const newProject = await createProject(payload);
    setProjects((prev) => [newProject, ...prev]);
    setSelectedProject(newProject);
  };

  return (
    <div className="app-root">
      <header className="app-navbar" id="app-navbar">
        <div className="nav-brand" onClick={() => setSelectedProject(null)} role="button" tabIndex={0}>
          <span className="brand-icon">🤖</span>
          <span className="brand-name">Dexmiq AI Agent</span>
          <span className="brand-badge">BRD Demo</span>
        </div>

        {selectedProject && (
          <div className="nav-project-crumb">
            <span className="crumb-sep">/</span>
            <span className="crumb-name">{selectedProject.name}</span>
          </div>
        )}
      </header>

      <div className="app-main-content">
        {!selectedProject ? (
          <ProjectList
            projects={projects}
            loading={loading}
            error={error}
            onSelectProject={(project) => setSelectedProject(project)}
            onCreateProject={handleCreateProject}
            onRefresh={fetchProjects}
          />
        ) : (
          <ProjectWorkspace
            project={selectedProject}
            onBackToProjects={() => setSelectedProject(null)}
          />
        )}
      </div>
    </div>
  );
};

export default App;
