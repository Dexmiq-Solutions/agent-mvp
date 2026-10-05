// =============================================================================
// Project Service — CRUD operations for /projects
// =============================================================================

import apiClient from './api';
import type {
  Project,
  CreateProjectPayload,
  UpdateProjectPayload,
  PaginatedResponse,
} from '../types';

const BASE = 'projects';

/** Fetch a paginated list of projects. */
export async function listProjects(
  limit = 20,
  offset = 0,
): Promise<PaginatedResponse<Project>> {
  // The backend GET /projects returns a plain Project[] array,
  // not a paginated wrapper. We receive the raw array and wrap it
  // into the PaginatedResponse shape consumed by the UI.
  const { data } = await apiClient.get<Project[]>(BASE, {
    params: { limit, offset },
  });
  const items = Array.isArray(data) ? data : [];
  return {
    items,
    total: items.length,
    limit,
    offset,
  };
}

/** Retrieve a single project by ID. */
export async function getProject(projectId: string): Promise<Project> {
  const { data } = await apiClient.get<Project>(`${BASE}/${projectId}`);
  return data;
}

/** Create a new project. */
export async function createProject(
  payload: CreateProjectPayload,
): Promise<Project> {
  const { data } = await apiClient.post<Project>(BASE, payload);
  return data;
}

/** Partially update a project (rename / edit description). */
export async function updateProject(
  projectId: string,
  payload: UpdateProjectPayload,
): Promise<Project> {
  const { data } = await apiClient.patch<Project>(
    `${BASE}/${projectId}`,
    payload,
  );
  return data;
}

/** Delete a project and all its associated data. */
export async function deleteProject(projectId: string): Promise<void> {
  await apiClient.delete(`${BASE}/${projectId}`);
}
