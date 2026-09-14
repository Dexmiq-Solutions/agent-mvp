"""Application service orchestrating project-scoped RAG + LLM generation."""

import time
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import Settings, get_settings
from observability.logging import get_logger
from exceptions.conversation import (
    ConversationMessageMismatchError,
    ConversationNotFoundError,
    ProjectConversationMismatchError,
)
from exceptions.generation import (
    GenerationError,
    LLMError,
    PromptConstructionError,
    RegenerationExhaustedError,
)
from exceptions.retrieval import RetrievalError
from models.message import MessageModel
from rag.generation.evaluation.models import EvaluationResult
from rag.generation.evaluation.service import (
    EvaluationService,
    get_evaluation_service,
)
from rag.generation.formatting.service import (
    ContextFormattingService,
    get_context_formatting_service,
)
from llm.models import LLMResult
from llm.service import LLMService, get_llm_service
from rag.generation.postprocessing.models import ProcessedResponse
from rag.generation.postprocessing.service import (
    PostProcessingService,
    get_post_processing_service,
)
from rag.generation.prompt.models import ConstructedPrompt
from rag.generation.prompt.service import (
    PromptConstructionService,
    get_prompt_construction_service,
)
from rag.retrieval.config import RetrievalConfig
from rag.retrieval.models import RetrievalResult
from rag.retrieval.service import RAGService
from schemas.generation import GenerationResult
from services.conversation_service import ConversationService
from services.rag_service import get_rag_service

logger = get_logger(__name__)

_generation_service: Optional["GenerationService"] = None


class GenerationService:
    """Application-level service orchestrating project-isolated RAG + LLM response generation.

    Coordinates the end-to-end generation workflow:
        User Message
             ↓
        Conversation & Project Validation
             ↓
        User Message Persistence (immediate commit)
             ↓
        RAG Retrieval Orchestration (dense + sparse + fusion + reranking + hydration + assembly)
             ↓
        Context Formatting (model-readable structured context)
             ↓
        Prompt Construction (instructions + context + authoritative original query)
             ↓
        LLM Interface (inference with bounded timeouts and retries)
             ↓
        Post-processing (deterministic normalization)
             ↓
        Groundedness & Safety Evaluation Quality Gate (bounded feedback regeneration)
             ↓
        Assistant Message Persistence (commit)
             ↓
        GenerationResult

    Single Source of Truth Guarantees:
    - LLM Configuration: Consumes existing Settings and LLMConfig.
    - Provider Execution: Consumes existing LLMService / BaseLLMInterface.
    - Retry & Timeout: Provider-level network retries handled by OpenAICompatibleLLMAdapter;
      semantic quality-gate regeneration handled by bounded evaluation loop.
    - Message Persistence: Consumes existing ConversationService.
    - Project Isolation: Validates project boundary at conversation, RAG, and prompt layers.
    - Transaction Boundaries: Releases database locks before external RAG/LLM network calls.
    """

    def __init__(
        self,
        session: AsyncSession,
        conversation_service: Optional[ConversationService] = None,
        rag_service: Optional[RAGService] = None,
        prompt_service: Optional[PromptConstructionService] = None,
        llm_service: Optional[LLMService] = None,
        post_processing_service: Optional[PostProcessingService] = None,
        evaluation_service: Optional[EvaluationService] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize GenerationService with required stage dependencies.

        Args:
            session: Active asynchronous SQLAlchemy database session.
            conversation_service: Conversation and Message lifecycle service.
            rag_service: Project-isolated RAG retrieval service.
            prompt_service: Prompt construction service.
            llm_service: LLM inference execution service.
            post_processing_service: Post-processing normalization service.
            evaluation_service: Groundedness and safety evaluation service.
            settings: Application settings single source of truth.
        """
        self._session = session
        self._settings = settings or get_settings()

        self._conversation_service = conversation_service or ConversationService(session=session)
        self._rag_service = rag_service or get_rag_service(settings=self._settings)
        self._prompt_service = prompt_service or get_prompt_construction_service(settings=self._settings)
        self._llm_service = llm_service or get_llm_service(settings=self._settings)
        self._post_processing_service = post_processing_service or get_post_processing_service(
            settings=self._settings
        )
        self._evaluation_service = evaluation_service or get_evaluation_service(settings=self._settings)

    @property
    def session(self) -> AsyncSession:
        """Return active database session."""
        return self._session

    @property
    def conversation_service(self) -> ConversationService:
        """Return conversation service instance."""
        return self._conversation_service

    @property
    def rag_service(self) -> RAGService:
        """Return RAG service instance."""
        return self._rag_service

    @property
    def llm_service(self) -> LLMService:
        """Return LLM service instance."""
        return self._llm_service

    async def generate_response(
        self,
        project_id: str,
        conversation_id: str,
        user_message: str | MessageModel,
        *,
        system_instruction: Optional[str] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        metadata: Optional[dict[str, Any]] = None,
        max_regeneration_attempts: Optional[int] = None,
        **llm_kwargs: Any,
    ) -> GenerationResult:
        """Execute project-isolated RAG-backed LLM generation for a user conversational turn.

        Args:
            project_id: Mandatory project boundary identifier.
            conversation_id: Target conversation identifier.
            user_message: Raw user query string or pre-persisted MessageModel turn.
            system_instruction: Optional system instruction override.
            retrieval_config: Optional per-request RetrievalConfig overrides.
            metadata: Optional metadata to attach to the user message turn.
            max_regeneration_attempts: Maximum regeneration attempts on quality gate failure.
            **llm_kwargs: Additional arguments passed to LLM execution.

        Returns:
            GenerationResult: Structured generation outcome containing accepted assistant response,
                retrieval telemetry, token usage, evaluation results, and persisted message models.

        Raises:
            ConversationNotFoundError: If conversation does not exist.
            ProjectConversationMismatchError: If conversation belongs to another project.
            RegenerationExhaustedError: If all allowed generation attempts fail quality gate.
            RetrievalError: If RAG retrieval pipeline fails.
            LLMError: If LLM provider execution fails.
            GenerationError: If pipeline execution fails unexpectedly.
        """
        overall_start_time = time.perf_counter()
        stage_latencies: dict[str, float] = {}

        # ----------------------------------------------------------------------
        # 1. Project Boundary & Conversation Existence Verification
        # ----------------------------------------------------------------------
        conversation = await self._conversation_service.get_conversation(
            project_id=project_id,
            conversation_id=conversation_id,
        )

        # ----------------------------------------------------------------------
        # 2. Persist / Resolve Authoritative User Message Turn
        # ----------------------------------------------------------------------
        user_msg_model: MessageModel
        if isinstance(user_message, MessageModel):
            if user_message.conversation_id != conversation_id:
                raise ConversationMessageMismatchError(
                    message_id=user_message.id,
                    expected_conversation_id=conversation_id,
                    actual_conversation_id=user_message.conversation_id,
                )
            user_msg_model = user_message
            original_user_query = user_message.content
        else:
            # Persist new user message turn
            t_user_save = time.perf_counter()
            user_msg_model = await self._conversation_service.create_message(
                project_id=project_id,
                conversation_id=conversation_id,
                content=str(user_message),
                role="user",
                metadata=metadata,
            )
            # Commit immediately so the user action is permanently stored
            # and database write locks are released before external calls
            await self._session.commit()
            stage_latencies["user_message_persistence_ms"] = (time.perf_counter() - t_user_save) * 1000.0
            original_user_query = user_msg_model.content

        logger.info(
            "Initiating RAG generation: project_id='%s', conversation_id='%s', user_message_id='%s', query_chars=%d",
            project_id,
            conversation_id,
            user_msg_model.id,
            len(original_user_query),
        )

        # ----------------------------------------------------------------------
        # 3. Project-Scoped RAG Retrieval Execution
        # ----------------------------------------------------------------------
        t_retrieval = time.perf_counter()
        try:
            retrieval_result: RetrievalResult = await self._rag_service.retrieve(
                project_id=project_id,
                query=original_user_query,
                config=retrieval_config,
                session=self._session,
            )
        except Exception as exc:
            logger.error(
                "RAG retrieval failed during generation for project '%s', conversation '%s': %s",
                project_id,
                conversation_id,
                exc,
                exc_info=True,
            )
            raise
        retrieval_duration_ms = (time.perf_counter() - t_retrieval) * 1000.0
        stage_latencies["retrieval_ms"] = retrieval_duration_ms

        logger.info(
            "RAG retrieval completed: project_id='%s', chunks_retrieved=%d, duration=%.2fms",
            project_id,
            len(retrieval_result.chunks),
            retrieval_duration_ms,
        )

        # Authoritative context produced by RAG Retrieval and Context Formatting stage
        retrieved_context = retrieval_result.formatted_context or retrieval_result.assembled_context

        # ----------------------------------------------------------------------
        # 4. Bounded Generation Loop (Prompt -> LLM -> Post-processing -> Evaluation)
        # ----------------------------------------------------------------------
        configured_max_regen = self._evaluation_service.config.max_regeneration_attempts
        effective_max_regen = (
            max_regeneration_attempts
            if max_regeneration_attempts is not None
            else configured_max_regen
        )
        total_attempts = 1 + max(0, effective_max_regen)

        evaluation_feedback: Optional[str] = None
        last_evaluation: Optional[EvaluationResult] = None
        last_processed: Optional[ProcessedResponse] = None
        accepted_processed: Optional[ProcessedResponse] = None
        generation_attempts_info: list[dict[str, Any]] = []

        for attempt in range(1, total_attempts + 1):
            attempt_t0 = time.perf_counter()
            is_regeneration = attempt > 1

            if is_regeneration:
                logger.info(
                    "Starting bounded generation regeneration attempt %d/%d for project '%s', conversation '%s'",
                    attempt,
                    total_attempts,
                    project_id,
                    conversation_id,
                )

            # 4a. Prompt Construction
            t_prompt = time.perf_counter()
            constructed_prompt: ConstructedPrompt = self._prompt_service.construct(
                query=original_user_query,
                context=retrieved_context,
                system_instruction=system_instruction,
                project_id=project_id,
                evaluation_feedback=evaluation_feedback,
            )
            prompt_ms = (time.perf_counter() - t_prompt) * 1000.0

            # 4b. LLM Inference Execution (via provider adapter with bounded timeout/retries)
            t_llm = time.perf_counter()
            try:
                llm_result: LLMResult = await self._llm_service.generate(
                    prompt=constructed_prompt,
                    **llm_kwargs,
                )
            except Exception as exc:
                logger.error(
                    "LLM generation failed on attempt %d for project '%s': %s",
                    attempt,
                    project_id,
                    exc,
                )
                raise
            llm_ms = (time.perf_counter() - t_llm) * 1000.0

            # 4c. Post-processing Normalization
            t_post = time.perf_counter()
            processed_response: ProcessedResponse = self._post_processing_service.process(llm_result)
            post_ms = (time.perf_counter() - t_post) * 1000.0
            last_processed = processed_response

            # 4d. Groundedness & Safety Evaluation Quality Gate
            t_eval = time.perf_counter()
            try:
                eval_result: EvaluationResult = await self._evaluation_service.evaluate_async(
                    query=original_user_query,
                    context=retrieved_context,
                    response=processed_response,
                    project_id=project_id,
                )

            except Exception as exc:
                logger.error(
                    "Evaluation quality gate failed on attempt %d for project '%s': %s",
                    attempt,
                    project_id,
                    exc,
                )
                raise
            eval_ms = (time.perf_counter() - t_eval) * 1000.0
            last_evaluation = eval_result

            attempt_duration_ms = (time.perf_counter() - attempt_t0) * 1000.0
            attempt_info = {
                "attempt": attempt,
                "is_regeneration": is_regeneration,
                "grounded": eval_result.grounded,
                "safe": eval_result.safe,
                "passed": eval_result.passed,
                "reason": eval_result.reason,
                "prompt_ms": round(prompt_ms, 2),
                "llm_ms": round(llm_ms, 2),
                "post_processing_ms": round(post_ms, 2),
                "evaluation_ms": round(eval_ms, 2),
                "attempt_duration_ms": round(attempt_duration_ms, 2),
            }
            generation_attempts_info.append(attempt_info)

            # 4e. Quality Gate Decision
            if eval_result.passed:
                logger.info(
                    "Generation passed quality gate on attempt %d/%d (grounded=%s, safe=%s, duration=%.2fms)",
                    attempt,
                    total_attempts,
                    eval_result.grounded,
                    eval_result.safe,
                    attempt_duration_ms,
                )
                accepted_processed = processed_response
                break

            logger.warning(
                "Generation attempt %d/%d rejected by quality gate (grounded=%s, safe=%s): '%s'",
                attempt,
                total_attempts,
                eval_result.grounded,
                eval_result.safe,
                eval_result.reason,
            )

            # Propagate evaluator critique as feedback for next bounded regeneration attempt
            if attempt < total_attempts:
                evaluation_feedback = eval_result.reason

        # ----------------------------------------------------------------------
        # 5. Handle Quality Gate Exhaustion
        # ----------------------------------------------------------------------
        if accepted_processed is None:
            total_duration_ms = (time.perf_counter() - overall_start_time) * 1000.0
            logger.error(
                "Generation rejected after %d attempts for project '%s', conversation '%s'. Total duration=%.2fms",
                total_attempts,
                project_id,
                conversation_id,
                total_duration_ms,
            )
            # Never fabricate a fake successful assistant response
            raise RegenerationExhaustedError(
                f"Generation failed groundedness/safety quality gate after {total_attempts} attempts. "
                f"Last evaluator rejection reason: {last_evaluation.reason if last_evaluation else 'Unknown'}",
                attempts=total_attempts,
                last_evaluation=last_evaluation,
                last_response=last_processed,
            )

        # ----------------------------------------------------------------------
        # 6. Persist Accepted Assistant Message Turn
        # ----------------------------------------------------------------------
        t_asst_save = time.perf_counter()
        total_duration_ms = (time.perf_counter() - overall_start_time) * 1000.0

        assistant_metadata: dict[str, Any] = {
            "user_message_id": user_msg_model.id,
            "finish_reason": accepted_processed.finish_reason,
            "usage": accepted_processed.usage.to_dict() if accepted_processed.usage else None,
            "model": accepted_processed.model or self._llm_service.config.model,
            "retrieval": {
                "chunks_count": len(retrieval_result.chunks),
                "duration_ms": round(retrieval_duration_ms, 2),
                "attempts_count": retrieval_result.execution_metadata.attempts_count,
                "retrieval_query": retrieval_result.retrieval_query,
                "fallback_triggered": retrieval_result.execution_metadata.fallback_triggered,
            },
            "evaluation": last_evaluation.to_dict() if last_evaluation else None,
            "evaluation_passed": True,
            "generation_attempts": len(generation_attempts_info),
            "total_duration_ms": round(total_duration_ms, 2),
        }

        assistant_msg_model = await self._conversation_service.create_message(
            project_id=project_id,
            conversation_id=conversation_id,
            content=accepted_processed.content,
            role="assistant",
            metadata=assistant_metadata,
        )
        await self._session.commit()
        stage_latencies["assistant_message_persistence_ms"] = (time.perf_counter() - t_asst_save) * 1000.0

        logger.info(
            "Generation workflow completed successfully: project_id='%s', conversation_id='%s', "
            "assistant_message_id='%s', total_duration=%.2fms",
            project_id,
            conversation_id,
            assistant_msg_model.id,
            total_duration_ms,
        )

        # ----------------------------------------------------------------------
        # 7. Construct Application-Level GenerationResult
        # ----------------------------------------------------------------------
        execution_metadata: dict[str, Any] = {
            "total_duration_ms": round(total_duration_ms, 2),
            "generation_attempts_count": len(generation_attempts_info),
            "regeneration_triggered": len(generation_attempts_info) > 1,
            "generation_attempts": generation_attempts_info,
            "model": accepted_processed.model or self._llm_service.config.model,
            "stage_latencies_ms": {k: round(v, 2) for k, v in stage_latencies.items()},
        }

        return GenerationResult(
            content=accepted_processed.content,
            finish_reason=accepted_processed.finish_reason,
            usage=accepted_processed.usage,
            project_id=project_id,
            conversation_id=conversation_id,
            user_message_id=user_msg_model.id,
            assistant_message_id=assistant_msg_model.id,
            retrieval_metadata=retrieval_result.execution_metadata.to_dict(),
            execution_metadata=execution_metadata,
            evaluation_metadata=last_evaluation.to_dict() if last_evaluation else {},
            user_message=user_msg_model,
            assistant_message=assistant_msg_model,
        )


def get_generation_service(
    session: AsyncSession,
    conversation_service: Optional[ConversationService] = None,
    rag_service: Optional[RAGService] = None,
    prompt_service: Optional[PromptConstructionService] = None,
    llm_service: Optional[LLMService] = None,
    post_processing_service: Optional[PostProcessingService] = None,
    evaluation_service: Optional[EvaluationService] = None,
    settings: Optional[Settings] = None,
) -> GenerationService:
    """Factory helper creating a configured GenerationService instance.

    Args:
        session: Active asynchronous SQLAlchemy session.
        conversation_service: Optional ConversationService override.
        rag_service: Optional RAGService override.
        prompt_service: Optional PromptConstructionService override.
        llm_service: Optional LLMService override.
        post_processing_service: Optional PostProcessingService override.
        evaluation_service: Optional EvaluationService override.
        settings: Optional Settings override.

    Returns:
        GenerationService: Configured application generation service.
    """
    return GenerationService(
        session=session,
        conversation_service=conversation_service,
        rag_service=rag_service,
        prompt_service=prompt_service,
        llm_service=llm_service,
        post_processing_service=post_processing_service,
        evaluation_service=evaluation_service,
        settings=settings,
    )


def reset_generation_service() -> None:
    """Reset cached singleton state (useful for test isolation)."""
    global _generation_service
    _generation_service = None
