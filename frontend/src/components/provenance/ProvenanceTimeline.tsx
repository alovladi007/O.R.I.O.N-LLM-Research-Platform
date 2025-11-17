'use client';

/**
 * ProvenanceTimeline Component
 *
 * Displays a vertical timeline of provenance events for an entity.
 * Shows the complete lifecycle with visual indicators, durations,
 * and expandable details for each event.
 */

import React, { useState } from 'react';
import {
  TimelineEvent,
  getEventColor,
  getEventIcon,
  formatDuration,
  formatTimestamp,
} from '@/types/provenance';

interface ProvenanceTimelineProps {
  events: TimelineEvent[];
  codeVersion?: string | null;
  totalDuration?: number | null;
  loading?: boolean;
  error?: string | null;
}

interface TimelineItemProps {
  event: TimelineEvent;
  isLast: boolean;
}

const TimelineItem: React.FC<TimelineItemProps> = ({ event, isLast }) => {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const colorClasses = {
    success: 'bg-green-500 border-green-600 text-green-900',
    error: 'bg-red-500 border-red-600 text-red-900',
    warning: 'bg-yellow-500 border-yellow-600 text-yellow-900',
    info: 'bg-blue-500 border-blue-600 text-blue-900',
  };

  const bgClasses = {
    success: 'bg-green-50 border-green-200',
    error: 'bg-red-50 border-red-200',
    warning: 'bg-yellow-50 border-yellow-200',
    info: 'bg-blue-50 border-blue-200',
  };

  const handleCopyDetails = () => {
    navigator.clipboard.writeText(JSON.stringify(event.details, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative pb-8">
      {/* Vertical line */}
      {!isLast && (
        <div className="absolute left-4 top-8 bottom-0 w-0.5 bg-gray-300" />
      )}

      <div className="relative flex items-start">
        {/* Event icon */}
        <div
          className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center border-2 ${
            colorClasses[event.color]
          } shadow-md z-10`}
        >
          <span className="text-sm">{getEventIcon(event.type)}</span>
        </div>

        {/* Event content */}
        <div className="ml-4 flex-grow">
          <div
            className={`rounded-lg border p-4 ${bgClasses[event.color]} cursor-pointer hover:shadow-md transition-shadow`}
            onClick={() => setExpanded(!expanded)}
          >
            {/* Header */}
            <div className="flex items-start justify-between">
              <div className="flex-grow">
                <h3 className="font-semibold text-gray-900">{event.title}</h3>
                <p className="text-sm text-gray-600 mt-1">{event.description}</p>
              </div>
              <div className="ml-4 text-right flex-shrink-0">
                <div className="text-xs text-gray-500">
                  {formatTimestamp(event.timestamp, {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </div>
                {event.duration_ms !== undefined && event.duration_ms !== null && (
                  <div className="text-xs text-gray-500 mt-1">
                    +{formatDuration(event.duration_ms)}
                  </div>
                )}
              </div>
            </div>

            {/* Expanded details */}
            {expanded && Object.keys(event.details).length > 0 && (
              <div className="mt-4 border-t pt-4">
                <div className="flex justify-between items-center mb-2">
                  <h4 className="text-sm font-medium text-gray-700">Details</h4>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCopyDetails();
                    }}
                    className="text-xs px-2 py-1 bg-white border border-gray-300 rounded hover:bg-gray-50 transition-colors"
                  >
                    {copied ? '✓ Copied' : '📋 Copy JSON'}
                  </button>
                </div>
                <div className="bg-white rounded border border-gray-200 p-3 max-h-64 overflow-y-auto">
                  <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono">
                    {JSON.stringify(event.details, null, 2)}
                  </pre>
                </div>
              </div>
            )}

            {/* Expand indicator */}
            {Object.keys(event.details).length > 0 && (
              <div className="mt-2 text-center">
                <span className="text-xs text-gray-500">
                  {expanded ? '▲ Click to collapse' : '▼ Click to expand details'}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const ProvenanceTimeline: React.FC<ProvenanceTimelineProps> = ({
  events,
  codeVersion,
  totalDuration,
  loading = false,
  error = null,
}) => {
  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-gray-600">Loading provenance data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
        <h3 className="font-semibold mb-2">Error Loading Provenance</h3>
        <p className="text-sm">{error}</p>
      </div>
    );
  }

  if (events.length === 0) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center text-gray-500">
        <p>No provenance records found for this entity.</p>
      </div>
    );
  }

  return (
    <div className="provenance-timeline">
      {/* Header with metadata */}
      <div className="mb-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
        <div className="flex justify-between items-center flex-wrap gap-4">
          <div>
            <h3 className="font-semibold text-gray-900">Provenance Timeline</h3>
            <p className="text-sm text-gray-600 mt-1">
              {events.length} event{events.length !== 1 ? 's' : ''} recorded
            </p>
          </div>
          <div className="flex gap-6 text-sm">
            {totalDuration !== null && totalDuration !== undefined && (
              <div>
                <span className="text-gray-500">Total Duration:</span>
                <span className="ml-2 font-medium text-gray-900">
                  {formatDuration(totalDuration)}
                </span>
              </div>
            )}
            {codeVersion && (
              <div>
                <span className="text-gray-500">Code Version:</span>
                <span className="ml-2 font-mono text-xs bg-gray-200 px-2 py-1 rounded">
                  {codeVersion}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Timeline events */}
      <div className="relative">
        {events.map((event, index) => (
          <TimelineItem
            key={`${event.type}-${event.timestamp}-${index}`}
            event={event}
            isLast={index === events.length - 1}
          />
        ))}
      </div>

      {/* Footer */}
      <div className="mt-6 p-4 bg-gray-50 border border-gray-200 rounded-lg text-center text-sm text-gray-500">
        <p>
          Timeline shows events from{' '}
          {formatTimestamp(events[0].timestamp, { dateStyle: 'medium', timeStyle: 'short' })}{' '}
          to{' '}
          {formatTimestamp(events[events.length - 1].timestamp, {
            dateStyle: 'medium',
            timeStyle: 'short',
          })}
        </p>
      </div>
    </div>
  );
};

export default ProvenanceTimeline;
