// =============================================================================
// TelemetryView — Collapsible assistant-message metadata panel
// =============================================================================

import React, { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import type { TelemetryMetadata } from '../../types';

interface TelemetryViewProps {
  metadata: TelemetryMetadata;
}

function MetaChip({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '4px',
        fontSize: '10.5px',
        color: '#606060',
        flexShrink: 0,
      }}
    >
      <span style={{ color: '#3e3e3e' }}>{label}</span>
      <span style={{ color: '#606060' }}>{value}</span>
    </span>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', minWidth: 0 }}>
      <span style={{ fontSize: '11px', color: '#444', width: '120px', flexShrink: 0, lineHeight: '1.5' }}>
        {label}
      </span>
      <span style={{ fontSize: '11px', color: '#777', lineHeight: '1.5', wordBreak: 'break-all', flex: 1 }}>
        {value}
      </span>
    </div>
  );
}

function StatusBadge({ ok, label }: { ok: boolean; label?: string }) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '3px',
        padding: '1px 6px',
        borderRadius: '4px',
        fontSize: '10.5px',
        fontWeight: 500,
        background: ok ? 'rgba(74,222,128,0.08)' : 'rgba(248,113,113,0.08)',
        color: ok ? '#4ade80' : '#f87171',
        border: `1px solid ${ok ? 'rgba(74,222,128,0.2)' : 'rgba(248,113,113,0.2)'}`,
      }}
    >
      <span
        style={{
          width: '5px',
          height: '5px',
          borderRadius: '50%',
          background: ok ? '#4ade80' : '#f87171',
          flexShrink: 0,
        }}
      />
      {label ?? (ok ? 'Yes' : 'No')}
    </span>
  );
}

export const TelemetryView: React.FC<TelemetryViewProps> = ({ metadata }) => {
  const [open, setOpen] = useState(false);

  // Summary fields
  const model = metadata.model;
  const latencyMs = metadata.total_duration_ms ?? metadata.processing_time_ms;
  const totalTokens = metadata.usage?.total_tokens;
  const chunksRetrieved =
    metadata.retrieval?.chunks_retrieved ??
    (metadata.retrieval_results ? metadata.retrieval_results.length : undefined);

  const hasSummary = !!(model || latencyMs !== undefined || totalTokens !== undefined || chunksRetrieved !== undefined);
  if (!hasSummary) return null;

  // Expanded-only fields
  const promptTokens = metadata.usage?.prompt_tokens;
  const completionTokens = metadata.usage?.completion_tokens;
  const retrievalQuery = metadata.retrieval?.query;
  const retrievalMs = metadata.retrieval?.duration_ms;
  const groundednessScore = metadata.groundedness_score;
  const grounded = metadata.grounded ?? metadata.evaluation_passed;
  const safety = metadata.safety;
  const evaluation = metadata.evaluation;
  const generationAttempts = metadata.generation_attempts;

  const hasExpanded = !!(
    promptTokens !== undefined ||
    completionTokens !== undefined ||
    retrievalQuery ||
    retrievalMs !== undefined ||
    groundednessScore !== undefined ||
    grounded !== undefined ||
    safety ||
    evaluation ||
    generationAttempts !== undefined
  );

  return (
    <div
      style={{
        marginTop: '12px',
        borderRadius: '10px',
        border: '1px solid #222',
        background: '#0d0d0d',
        overflow: 'hidden',
      }}
    >
      {/* Summary row / toggle */}
      <button
        onClick={() => setOpen((v) => !v)}
        disabled={!hasExpanded}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          padding: '8px 14px',
          background: 'transparent',
          border: 'none',
          cursor: hasExpanded ? 'pointer' : 'default',
          textAlign: 'left',
        }}
        onMouseEnter={(e) => {
          if (hasExpanded) (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.02)';
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
        }}
        aria-expanded={open}
        aria-label="Toggle telemetry details"
      >
        <span
          style={{
            fontSize: '9.5px',
            fontWeight: 600,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: '#333',
            flexShrink: 0,
          }}
        >
          Telemetry
        </span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flex: 1, minWidth: 0, flexWrap: 'wrap' }}>
          {/* Divider */}
          <div style={{ width: '1px', height: '10px', background: '#222', flexShrink: 0 }} />

          {model && <MetaChip label="Model" value={model} />}
          {totalTokens !== undefined && <MetaChip label="Tokens" value={totalTokens.toLocaleString()} />}
          {latencyMs !== undefined && <MetaChip label="Latency" value={`${(latencyMs / 1000).toFixed(2)}s`} />}
          {chunksRetrieved !== undefined && <MetaChip label="Chunks" value={chunksRetrieved} />}
        </div>

        {hasExpanded && (
          <span style={{ color: open ? '#555' : '#333', transition: 'color 0.12s ease', flexShrink: 0 }}>
            {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </span>
        )}
      </button>

      {/* Expanded rows */}
      {open && hasExpanded && (
        <div
          style={{
            borderTop: '1px solid #1a1a1a',
            padding: '10px 12px',
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
          }}
        >
          {promptTokens !== undefined && <Row label="Prompt tokens" value={promptTokens.toLocaleString()} />}
          {completionTokens !== undefined && <Row label="Completion tokens" value={completionTokens.toLocaleString()} />}
          {retrievalMs !== undefined && <Row label="Retrieval time" value={`${retrievalMs}ms`} />}
          {retrievalQuery && (
            <Row
              label="Retrieval query"
              value={<span style={{ fontStyle: 'italic', opacity: 0.7 }}>{retrievalQuery}</span>}
            />
          )}
          {groundednessScore !== undefined && (
            <Row label="Groundedness" value={`${(groundednessScore * 100).toFixed(0)}%`} />
          )}
          {grounded !== undefined && (
            <Row label="Grounded" value={<StatusBadge ok={grounded} />} />
          )}
          {safety && (
            <Row label="Safety" value={<StatusBadge ok={safety === 'safe'} label={safety} />} />
          )}
          {evaluation && (
            <Row label="Evaluation" value={<StatusBadge ok={evaluation === 'passed'} label={evaluation} />} />
          )}
          {generationAttempts !== undefined && <Row label="Gen. attempts" value={generationAttempts} />}
        </div>
      )}
    </div>
  );
};
