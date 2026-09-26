import { apiRequest } from './client.ts';
import type { SourceDocument } from '../types/index.ts';

export async function listSources(projectId: string): Promise<SourceDocument[]> {
  return apiRequest<SourceDocument[]>(`/projects/${projectId}/sources`);
}

export async function uploadSource(
  projectId: string,
  file: File,
  name?: string
): Promise<SourceDocument> {
  const formData = new FormData();
  formData.append('file', file);
  if (name && name.trim()) {
    formData.append('name', name.trim());
  }

  return apiRequest<SourceDocument>(`/projects/${projectId}/sources`, {
    method: 'POST',
    body: formData,
  });
}
