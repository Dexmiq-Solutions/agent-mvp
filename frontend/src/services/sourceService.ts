// =============================================================================
// Source Service — CRUD + upload operations for project sources (knowledge base)
// =============================================================================

import apiClient from './api';
import type {
  SourceDocument,
  UpdateSourcePayload,
  DocumentVersion,
  PaginatedResponse,
} from '../types';

function base(projectId: string) {
  return `projects/${projectId}/sources`;
}

/** Fetch all source documents for a project. */
export async function listSources(
  projectId: string,
  limit = 50,
  offset = 0,
): Promise<PaginatedResponse<SourceDocument>> {
  const { data } = await apiClient.get<PaginatedResponse<SourceDocument>>(
    base(projectId),
    { params: { limit, offset } },
  );
  return data;
}

/** Retrieve a single source document by ID. */
export async function getSource(
  projectId: string,
  sourceId: string,
): Promise<SourceDocument> {
  const { data } = await apiClient.get<SourceDocument>(
    `${base(projectId)}/${sourceId}`,
  );
  return data;
}

/**
 * Upload a new source document (multipart file upload).
 *
 * Accepted types: .md, .txt, .docx  (PDF is NOT supported by the backend).
 * The backend will auto-process the document through the RAG pipeline
 * when AUTO_PROCESS_DOCUMENTS is enabled.
 */
export async function uploadSource(
  projectId: string,
  file: File,
): Promise<SourceDocument> {
  const formData = new FormData();
  formData.append('file', file);

  const { data } = await apiClient.post<SourceDocument>(
    base(projectId),
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      // Long timeout for large file uploads + processing
      timeout: 120_000,
    },
  );
  return data;
}

/** Rename a source document. */
export async function updateSource(
  projectId: string,
  sourceId: string,
  payload: UpdateSourcePayload,
): Promise<SourceDocument> {
  const { data } = await apiClient.patch<SourceDocument>(
    `${base(projectId)}/${sourceId}`,
    payload,
  );
  return data;
}

/** Delete a source document and all its versions / vectors. */
export async function deleteSource(
  projectId: string,
  sourceId: string,
): Promise<void> {
  await apiClient.delete(`${base(projectId)}/${sourceId}`);
}

/**
 * Upload a new version of an existing source document.
 * Follows the backend's versioning endpoint.
 */
export async function uploadSourceVersion(
  projectId: string,
  sourceId: string,
  file: File,
): Promise<DocumentVersion> {
  const formData = new FormData();
  formData.append('file', file);

  const { data } = await apiClient.post<DocumentVersion>(
    `${base(projectId)}/${sourceId}/versions`,
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120_000,
    },
  );
  return data;
}
