/**
 * Provenance tracking types for NANO-OS
 *
 * These types define the structure of provenance data for tracking
 * the complete lifecycle and lineage of entities in the system.
 */

export type EntityType = 'JOB' | 'PREDICTION' | 'STRUCTURE' | 'MATERIAL' | 'WORKFLOW' | 'USER';

export type EventType =
  | 'CREATED'
  | 'QUEUED'
  | 'STARTED'
  | 'COMPLETED'
  | 'FAILED'
  | 'CANCELLED'
  | 'TIMEOUT'
  | 'PREDICTED'
  | 'UPLOADED'
  | 'MODIFIED'
  | 'DELETED'
  | 'ACCESSED'
  | 'VALIDATED';

export type EventColor = 'success' | 'error' | 'warning' | 'info';

export interface ProvenanceRecord {
  id: string;
  entity_type: EntityType;
  entity_id: string;
  event_type: EventType;
  timestamp: string;
  details: Record<string, any>;
  created_at: string;
}

export interface ProvenanceChain {
  entity_type: EntityType;
  entity_id: string;
  records: ProvenanceRecord[];
  total_records: number;
  first_event: string | null;
  last_event: string | null;
  duration_ms: number | null;
}

export interface TimelineEvent {
  type: EventType;
  timestamp: string;
  title: string;
  description: string;
  color: EventColor;
  details: Record<string, any>;
  duration_ms?: number;
}

export interface ProvenanceTimeline {
  entity_type: EntityType;
  entity_id: string;
  events: TimelineEvent[];
  total_duration_ms: number | null;
  code_version: string | null;
}

export interface ProvenanceSummary {
  entity_type: EntityType;
  entity_id: string;
  total_records: number;
  first_event: string | null;
  last_event: string | null;
  event_types: EventType[];
  duration_ms: number | null;
}

/**
 * Helper function to get color for event type
 */
export function getEventColor(eventType: EventType): EventColor {
  switch (eventType) {
    case 'COMPLETED':
    case 'PREDICTED':
      return 'success';
    case 'FAILED':
    case 'TIMEOUT':
    case 'DELETED':
      return 'error';
    case 'CANCELLED':
    case 'MODIFIED':
      return 'warning';
    case 'CREATED':
    case 'QUEUED':
    case 'STARTED':
    case 'UPLOADED':
    case 'ACCESSED':
    case 'VALIDATED':
    default:
      return 'info';
  }
}

/**
 * Helper function to get icon for event type
 */
export function getEventIcon(eventType: EventType): string {
  switch (eventType) {
    case 'CREATED':
      return '✨';
    case 'QUEUED':
      return '📋';
    case 'STARTED':
      return '▶️';
    case 'COMPLETED':
      return '✅';
    case 'FAILED':
      return '❌';
    case 'CANCELLED':
      return '🚫';
    case 'TIMEOUT':
      return '⏱️';
    case 'PREDICTED':
      return '🔮';
    case 'UPLOADED':
      return '📤';
    case 'MODIFIED':
      return '✏️';
    case 'DELETED':
      return '🗑️';
    case 'ACCESSED':
      return '👁️';
    case 'VALIDATED':
      return '✓';
    default:
      return '•';
  }
}

/**
 * Helper function to format duration
 */
export function formatDuration(durationMs: number): string {
  if (durationMs < 1000) {
    return `${Math.round(durationMs)}ms`;
  } else if (durationMs < 60000) {
    return `${(durationMs / 1000).toFixed(1)}s`;
  } else if (durationMs < 3600000) {
    const minutes = Math.floor(durationMs / 60000);
    const seconds = Math.floor((durationMs % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  } else {
    const hours = Math.floor(durationMs / 3600000);
    const minutes = Math.floor((durationMs % 3600000) / 60000);
    return `${hours}h ${minutes}m`;
  }
}

/**
 * Helper function to parse timestamp
 */
export function parseTimestamp(timestamp: string): Date {
  return new Date(timestamp);
}

/**
 * Helper function to format timestamp
 */
export function formatTimestamp(timestamp: string, options?: Intl.DateTimeFormatOptions): string {
  const date = parseTimestamp(timestamp);
  return date.toLocaleString(undefined, options || {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
}
