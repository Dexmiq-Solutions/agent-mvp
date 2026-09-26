import { apiRequest } from './client.ts';
import type { Project, ProjectCreatePayload } from '../types/index.ts';

export async function listProjects(): Promise<Project[]> {
  return apiRequest<Project[]>('/projects');
}

export async function createProject(payload: ProjectCreatePayload): Promise<Project> {
  return apiRequest<Project>('/projects', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function getProject(projectId: string): Promise<Project> {
  return apiRequest<Project>(`/projects/${projectId}`);
}
