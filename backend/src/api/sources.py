"""FastAPI router for project-scoped Document (Source) and Version operations."""

from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Query,
    Response,
    UploadFile,
    status,
)

from api.dependencies import get_document_service
from models.document import DocumentModel
from schemas.document import (
    DocumentDetailResponse,
    DocumentResponse,
    DocumentUpdate,
    DocumentVersionResponse,
)
from services.document_service import DocumentService

router = APIRouter(prefix="/projects/{project_id}/sources", tags=["Sources"])


def _to_document_response(doc: DocumentModel) -> DocumentResponse:
    """Helper to convert a DocumentModel to a DocumentResponse schema."""
    latest_version = doc.versions[-1] if doc.versions else None
    return DocumentResponse(
        id=doc.id,
        project_id=doc.project_id,
        name=doc.name,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        latest_version=(
            DocumentVersionResponse.model_validate(latest_version)
            if latest_version is not None
            else None
        ),
        versions_count=len(doc.versions),
    )


def _to_document_detail_response(doc: DocumentModel) -> DocumentDetailResponse:
    """Helper to convert a DocumentModel to a DocumentDetailResponse schema."""
    latest_version = doc.versions[-1] if doc.versions else None
    return DocumentDetailResponse(
        id=doc.id,
        project_id=doc.project_id,
        name=doc.name,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        latest_version=(
            DocumentVersionResponse.model_validate(latest_version)
            if latest_version is not None
            else None
        ),
        versions_count=len(doc.versions),
        versions=[DocumentVersionResponse.model_validate(v) for v in doc.versions],
    )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentResponse,
    summary="Upload a new source document",
)
async def create_source(
    project_id: str,
    file: UploadFile = File(..., description="Original source file binary"),
    name: Optional[str] = Form(None, description="Optional custom source display name"),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Upload a source document to a project, store in object storage, and create version 1."""
    content = await file.read()
    filename = file.filename or "uploaded_source"
    document = await service.create_document(
        project_id=project_id,
        filename=filename,
        file_data=content,
        content_type=file.content_type,
        name=name,
    )
    return _to_document_response(document)


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    response_model=list[DocumentResponse],
    summary="List all source documents for a project",
)
async def list_sources(
    project_id: str,
    limit: int = Query(100, ge=1, le=100, description="Maximum documents to return"),
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
    service: DocumentService = Depends(get_document_service),
) -> list[DocumentResponse]:
    """Retrieve all documents belonging strictly to the specified project boundary."""
    documents = await service.list_documents(
        project_id=project_id,
        limit=limit,
        offset=offset,
    )
    return [_to_document_response(doc) for doc in documents]


@router.get(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    response_model=DocumentDetailResponse,
    summary="Get source document details",
)
async def get_source(
    project_id: str,
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentDetailResponse:
    """Retrieve a document and all its versions under project isolation."""
    document = await service.get_document(
        project_id=project_id,
        document_id=document_id,
    )
    return _to_document_detail_response(document)


@router.patch(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    response_model=DocumentResponse,
    summary="Update source document metadata",
)
async def update_source(
    project_id: str,
    document_id: str,
    payload: DocumentUpdate,
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Update mutable metadata of a document under project isolation."""
    document = await service.update_document(
        project_id=project_id,
        document_id=document_id,
        name=payload.name,
    )
    return _to_document_response(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a source document",
)
async def delete_source(
    project_id: str,
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> Response:
    """Delete a document, its database cascade records, and its physical files in storage."""
    await service.delete_document(
        project_id=project_id,
        document_id=document_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{document_id}/versions",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentVersionResponse,
    summary="Upload a new version of an existing source document",
)
async def create_source_version(
    project_id: str,
    document_id: str,
    file: UploadFile = File(..., description="New version source file binary"),
    service: DocumentService = Depends(get_document_service),
) -> DocumentVersionResponse:
    """Upload a new physical file version for an existing document under project isolation."""
    content = await file.read()
    filename = file.filename or "updated_source"
    version = await service.create_document_version(
        project_id=project_id,
        document_id=document_id,
        filename=filename,
        file_data=content,
        content_type=file.content_type,
    )
    return DocumentVersionResponse.model_validate(version)
