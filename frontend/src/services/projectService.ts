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
  const { data } = await apiClient.get<PaginatedResponse<Project>>(BASE, {
    params: { limit, offset },
  });
  return data;
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
