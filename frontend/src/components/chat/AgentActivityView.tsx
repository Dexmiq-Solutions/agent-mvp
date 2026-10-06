import React, { useState, useMemo } from 'react';
import {
  Bot,
  Sparkles,
  ChevronDown,
  ChevronUp,
  Check,
  ArrowRight,
  AlertCircle,
  Clock,
  Layers,
  FileText,
  Database,
  Search,
} from 'lucide-react';
import type { AgentActivityItem, WorkflowProgressEvent } from '../../types';

export interface AgentActivityViewProps {
  /** The current active progress event from SSE (if still generating) */
  currentProgress?: WorkflowProgressEvent | null;
  /** Chronological list of recorded execution steps */
  activityLog?: AgentActivityItem[];
  /** Whether the agent is currently actively running/streaming */
  isLive?: boolean;
  /** Default collapsed state (default: false when live, true when completed) */
  defaultExpanded?: boolean;
  /** Optional custom CSS classes */
  className?: string;
}

/**
 * Derives a human-friendly, prominent headline for the current Agent state.
 */
function getProgressHeadline(progress?: WorkflowProgressEvent | null, isLive?: boolean): string {
  if (!progress && isLive) {
    return 'Agent Initiating';
  }
  if (!progress) {
    return 'Agent Execution Activity';
  }

  // If a specific section is present, headline highlights the section
  if (progress.section) {
    return `Working on ${progress.section}`;
  }

  // Fallback to phase-specific descriptive headline
  switch (progress.phase) {
    case '1_INITIAL_CONTEXT':
      return 'Analyzing Project Context & Scope';
    case '2_ACTION_DECISION':
      return 'Determining Execution Strategy';
    case '3_ACTION_EXECUTION':
      return 'Executing Knowledge & Task Actions';
    case '4_EVIDENCE_EVALUATION':
      return 'Evaluating Evidence Sufficiency';
    case '5_SECTION_ITERATION':
      return 'Drafting & Validating Requirements';
    case '6_DOCUMENT_ASSEMBLY':
      return 'Assembling Business Requirements Document';
    case '7_FINAL_VALIDATION':
      return 'Executing Document Validation';
    case '8_FINAL_RECOVERY':
      return 'Executing Final Validation Recovery';
    case '9_COMPLETION':
      return 'BRD Workflow Completed';
    default:
      if (progress.message) {
        if (progress.message.toLowerCase().includes('section')) {
          return 'Processing BRD Section';
        }
        if (progress.message.toLowerCase().includes('evidence') || progress.message.toLowerCase().includes('retriev')) {
          return 'Retrieving Project Knowledge';
        }
      }
      return isLive ? 'Agent in Progress' : 'Agent Activity Log';
  }
}

/**
 * Returns a contextual icon for the step based on message content and phase.
 */
function getStepCategoryIcon(item: AgentActivityItem) {
  const text = (item.message + ' ' + (item.phase || '')).toLowerCase();
  if (text.includes('retriev') || text.includes('knowledge') || text.includes('rag') || text.includes('evidence')) {
    return <Database size={12} className="text-indigo-400 flex-shrink-0" />;
  }
  if (text.includes('evaluat') || text.includes('analyz') || text.includes('validat')) {
    return <Search size={12} className="text-amber-400 flex-shrink-0" />;
  }
  if (text.includes('section') || text.includes('draft') || text.includes('assembl') || text.includes('brd')) {
    return <FileText size={12} className="text-blue-400 flex-shrink-0" />;
  }
  return <Layers size={12} className="text-purple-400 flex-shrink-0" />;
}

export const AgentActivityView: React.FC<AgentActivityViewProps> = ({
  currentProgress,
  activityLog = [],
  isLive = false,
  defaultExpanded,
  className = '',
}) => {
  // If defaultExpanded is explicitly set, use it; otherwise live defaults to open, completed defaults to closed
  const [isExpanded, setIsExpanded] = useState<boolean>(
    defaultExpanded !== undefined ? defaultExpanded : isLive
  );

  const headline = useMemo(
    () => getProgressHeadline(currentProgress, isLive),
    [currentProgress, isLive]
  );

  const liveSubtitle = useMemo(() => {
    if (currentProgress?.message) {
      return currentProgress.message;
    }
    if (isLive) {
      return 'Analyzing business requirements and project evidence...';
    }
    return null;
  }, [currentProgress, isLive]);

  const completedStepsCount = useMemo(
    () => activityLog.filter((item) => item.status === 'completed').length,
    [activityLog]
  );

  // If there are no activity items and not live, don't render empty box
  if (!isLive && activityLog.length === 0) {
    return null;
  }

  return (
    <div
      className={`rounded-xl border transition-all duration-200 overflow-hidden ${
        isLive
          ? 'bg-[#121217]/95 border-indigo-500/30 shadow-[0_4px_24px_rgba(99,102,241,0.08)]'
          : 'bg-[#141416]/90 border-zinc-800/80 hover:border-zinc-700/70'
      } ${className}`}
      id={isLive ? 'agent-live-activity' : 'agent-completed-activity'}
    >
      {/* ── Live Status Card (displayed when active) ─────────────────────────── */}
      {isLive && (
        <div className="p-3.5 sm:p-4 border-b border-indigo-500/15 bg-gradient-to-r from-indigo-950/20 via-zinc-900/40 to-transparent">
          <div className="flex items-start gap-3">
            {/* Animated Indicator */}
            <div className="relative flex items-center justify-center w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/25 flex-shrink-0 mt-0.5">
              <span className="absolute w-3.5 h-3.5 rounded-full bg-indigo-400/40 animate-ping" />
              <Sparkles size={15} className="text-indigo-400 animate-pulse" />
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap mb-1">
                <span className="text-[13.5px] font-semibold text-white tracking-tight">
                  {headline}
                </span>

                {currentProgress?.section && (
                  <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 max-w-[280px] truncate">
                    {currentProgress.section}
                  </span>
                )}

                {currentProgress?.phase && !currentProgress.section && (
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono uppercase bg-zinc-800 text-zinc-400 border border-zinc-700">
                    {currentProgress.phase.replace(/^\d+_/, '')}
                  </span>
                )}
              </div>

              {liveSubtitle && (
                <div className="flex items-center gap-2 text-[12.5px] text-zinc-400 leading-snug">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse flex-shrink-0" />
                  <p className="truncate">{liveSubtitle}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Collapsible Accordion Header ───────────────────────────────────── */}
      <div className="flex items-center justify-between px-3.5 py-2.5 bg-[#161619]/60">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 text-zinc-300">
            <Bot size={13} className="text-indigo-400" />
            <span className="text-[12px] font-semibold tracking-wide uppercase text-zinc-300">
              Agent Activity
            </span>
          </div>

          {activityLog.length > 0 && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium bg-zinc-800 text-zinc-400 border border-zinc-700/60">
              {isLive ? (
                <>
                  <span className="text-indigo-300 font-semibold mr-1">
                    {completedStepsCount}
                  </span>
                  / {activityLog.length} completed
                </>
              ) : (
                `${activityLog.length} steps`
              )}
            </span>
          )}
        </div>

        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-1 px-2.5 py-1 rounded-md text-[11.5px] font-medium text-zinc-400 hover:text-white hover:bg-zinc-800/60 transition-colors"
          aria-expanded={isExpanded}
          aria-label={isExpanded ? 'Hide agent activity log' : 'View agent activity log'}
          id="btn-toggle-agent-activity"
        >
          <span>{isExpanded ? 'Hide activity' : 'View activity'}</span>
          {isExpanded ? (
            <ChevronUp size={13} className="text-zinc-400" />
          ) : (
            <ChevronDown size={13} className="text-zinc-400" />
          )}
        </button>
      </div>

      {/* ── Collapsible Activity Steps Log ──────────────────────────────────── */}
      {isExpanded && (
        <div className="px-4 py-3 bg-[#111114]/90 border-t border-zinc-800/60 max-h-[360px] overflow-y-auto">
          {activityLog.length === 0 ? (
            <div className="flex items-center gap-2 py-2 text-[12px] text-zinc-500 italic">
              <Clock size={12} />
              <span>Waiting for initial execution events...</span>
            </div>
          ) : (
            <div className="relative pl-3 border-l border-zinc-800/90 space-y-2.5 my-1">
              {activityLog.map((item, index) => {
                const isCompleted = item.status === 'completed';
                const isInProgress = item.status === 'in_progress';
                const isFailed = item.status === 'failed';

                return (
                  <div
                    key={item.id || index}
                    className="relative flex items-start gap-2.5 text-[12.5px] group"
                  >
                    {/* Node Dot / Status Icon */}
                    <div
                      className={`absolute -left-[19px] top-0.5 w-4 h-4 rounded-full flex items-center justify-center border text-[9px] ${
                        isCompleted
                          ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-400'
                          : isInProgress
                          ? 'bg-indigo-500/20 border-indigo-400 text-indigo-300 ring-2 ring-indigo-500/20 animate-pulse'
                          : isFailed
                          ? 'bg-rose-500/20 border-rose-500 text-rose-400'
                          : 'bg-zinc-800 border-zinc-700 text-zinc-400'
                      }`}
                    >
                      {isCompleted && <Check size={10} strokeWidth={2.5} />}
                      {isInProgress && <ArrowRight size={10} strokeWidth={2.5} />}
                      {isFailed && <AlertCircle size={10} strokeWidth={2.5} />}
                    </div>

                    {/* Step Content */}
                    <div className="flex-1 min-w-0 flex items-baseline justify-between gap-3">
                      <div className="flex items-center gap-1.5 flex-wrap min-w-0">
                        {getStepCategoryIcon(item)}
                        <span
                          className={`leading-relaxed ${
                            isInProgress
                              ? 'text-indigo-200 font-medium'
                              : isFailed
                              ? 'text-rose-300'
                              : 'text-zinc-300'
                          }`}
                        >
                          {item.message}
                        </span>

                        {item.section && (
                          <span className="inline-block px-1.5 py-0.2 rounded text-[10.5px] font-medium bg-zinc-800/80 text-zinc-400 border border-zinc-700/50">
                            {item.section}
                          </span>
                        )}
                      </div>

                      {item.timestamp && (
                        <span className="text-[10px] text-zinc-500 font-mono whitespace-nowrap opacity-60 group-hover:opacity-100 transition-opacity">
                          {new Date(item.timestamp).toLocaleTimeString([], {
                            hour: '2-digit',
                            minute: '2-digit',
                            second: '2-digit',
                          })}
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
