"""FastAPI router for Project lifecycle management endpoints."""

from fastapi import APIRouter, Depends, Query, Response, status

from api.dependencies import get_project_service
from schemas.project import ProjectCreate, ProjectResponse, ProjectUpdate
from services.project_service import ProjectService

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=ProjectResponse,
    summary="Create a new project",
)
async def create_project(
    payload: ProjectCreate,
    service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    """Create a new root tenant Project."""
    project = await service.create_project(
        name=payload.name,
        description=payload.description,
    )
    return ProjectResponse.model_validate(project)


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    response_model=list[ProjectResponse],
    summary="List all projects",
)
async def list_projects(
    limit: int = Query(100, ge=1, le=100, description="Maximum number of projects to return"),
    offset: int = Query(0, ge=0, description="Number of projects to skip"),
    service: ProjectService = Depends(get_project_service),
) -> list[ProjectResponse]:
    """List all projects ordered by creation date descending."""
    projects = await service.list_projects(limit=limit, offset=offset)
    return [ProjectResponse.model_validate(p) for p in projects]


@router.get(
    "/{project_id}",
    status_code=status.HTTP_200_OK,
    response_model=ProjectResponse,
    summary="Get project by ID",
)
async def get_project(
    project_id: str,
    service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    """Retrieve details for a specific project."""
    project = await service.get_project(project_id=project_id)
    return ProjectResponse.model_validate(project)


@router.patch(
    "/{project_id}",
    status_code=status.HTTP_200_OK,
    response_model=ProjectResponse,
    summary="Update project details",
)
async def update_project(
    project_id: str,
    payload: ProjectUpdate,
    service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    """Update mutable fields of a project."""
    project = await service.update_project(
        project_id=project_id,
        name=payload.name,
        description=payload.description,
    )
    return ProjectResponse.model_validate(project)


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a project",
)
async def delete_project(
    project_id: str,
    service: ProjectService = Depends(get_project_service),
) -> Response:
    """Delete a project and cascade deletion to all related entities and storage files."""
    await service.delete_project(project_id=project_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
