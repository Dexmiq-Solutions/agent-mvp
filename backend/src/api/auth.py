"""FastAPI router for user authentication, registration, token refresh, and session management."""

from fastapi import APIRouter, Depends, Request, Response, status

from api.dependencies import get_auth_service, get_current_user
from models.user import UserModel
from schemas.auth import (
    RefreshTokenRequest,
    TokenResponse,
    UserLoginRequest,
    UserResponse,
    UserSignupRequest,
)
from services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _extract_client_ip(request: Request) -> str:
    """Extract client IP address accounting for proxies."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "127.0.0.1"


@router.post(
    "/signup",
    status_code=status.HTTP_201_CREATED,
    response_model=UserResponse,
    summary="Register a new user account",
)
async def signup(
    payload: UserSignupRequest,
    service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """Create a new user account with email and password."""
    user = await service.signup(
        email=payload.email,
        password=payload.password,
    )
    return UserResponse.model_validate(user)


@router.post(
    "/login",
    status_code=status.HTTP_200_OK,
    response_model=TokenResponse,
    summary="Authenticate user and issue tokens",
)
async def login(
    payload: UserLoginRequest,
    request: Request,
    service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """Authenticate with email and password to receive access and refresh tokens."""
    client_ip = _extract_client_ip(request)
    _, access_token, refresh_token, expires_in = await service.login(
        email=payload.email,
        password=payload.password,
        client_ip=client_ip,
    )
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post(
    "/refresh",
    status_code=status.HTTP_200_OK,
    response_model=TokenResponse,
    summary="Rotate refresh token and issue new token pair",
)
async def refresh_tokens(
    payload: RefreshTokenRequest,
    service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """Submit an active refresh token to receive a fresh access and refresh token pair."""
    _, access_token, refresh_token, expires_in = await service.refresh_tokens(
        refresh_token=payload.refresh_token,
    )
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke refresh token session",
)
async def logout(
    payload: RefreshTokenRequest,
    service: AuthService = Depends(get_auth_service),
) -> Response:
    """Revoke the presented refresh token."""
    await service.logout(refresh_token=payload.refresh_token)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/me",
    status_code=status.HTTP_200_OK,
    response_model=UserResponse,
    summary="Get current authenticated user profile",
)
async def get_current_user_profile(
    current_user: UserModel = Depends(get_current_user),
) -> UserResponse:
    """Retrieve profile details of the currently authenticated user."""
    return UserResponse.model_validate(current_user)
