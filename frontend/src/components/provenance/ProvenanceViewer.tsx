'use client';

/**
 * ProvenanceViewer Component
 *
 * Complete viewer for entity provenance with tabs for:
 * - Timeline view (visual timeline of events)
 * - Table view (detailed list of all events)
 * - Summary view (key metrics and statistics)
 *
 * This component can be integrated into any entity detail page
 * (jobs, predictions, structures, etc.) to show complete audit trails.
 */

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getProvenanceTimeline, getProvenanceSummary } from '@/lib/api';
import ProvenanceTimeline from './ProvenanceTimeline';
import {
  ProvenanceTimeline as ProvenanceTimelineType,
  ProvenanceSummary,
  formatDuration,
  formatTimestamp,
} from '@/types/provenance';

interface ProvenanceViewerProps {
  entityType: string;
  entityId: string;
}

type TabType = 'timeline' | 'summary';

const ProvenanceViewer: React.FC<ProvenanceViewerProps> = ({ entityType, entityId }) => {
  const [activeTab, setActiveTab] = useState<TabType>('timeline');

  // Fetch provenance timeline
  const {
    data: timelineData,
    isLoading: timelineLoading,
    error: timelineError,
  } = useQuery<ProvenanceTimelineType>({
    queryKey: ['provenance', 'timeline', entityType, entityId],
    queryFn: () => getProvenanceTimeline(entityType, entityId),
    enabled: !!entityType && !!entityId,
  });

  // Fetch provenance summary
  const {
    data: summaryData,
    isLoading: summaryLoading,
    error: summaryError,
  } = useQuery<ProvenanceSummary>({
    queryKey: ['provenance', 'summary', entityType, entityId],
    queryFn: () => getProvenanceSummary(entityType, entityId),
    enabled: !!entityType && !!entityId && activeTab === 'summary',
  });

  const tabs: { id: TabType; label: string }[] = [
    { id: 'timeline', label: 'Timeline' },
    { id: 'summary', label: 'Summary' },
  ];

  return (
    <div className="provenance-viewer">
      {/* Tab Navigation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                py-4 px-1 border-b-2 font-medium text-sm transition-colors
                ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }
              `}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'timeline' && (
          <ProvenanceTimeline
            events={timelineData?.events || []}
            codeVersion={timelineData?.code_version}
            totalDuration={timelineData?.total_duration_ms}
            loading={timelineLoading}
            error={timelineError ? String(timelineError) : null}
          />
        )}

        {activeTab === 'summary' && (
          <div>
            {summaryLoading ? (
              <div className="flex items-center justify-center p-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                <span className="ml-3 text-gray-600">Loading summary...</span>
              </div>
            ) : summaryError ? (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
                <h3 className="font-semibold mb-2">Error Loading Summary</h3>
                <p className="text-sm">{String(summaryError)}</p>
              </div>
            ) : summaryData ? (
              <div className="space-y-6">
                {/* Summary Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <SummaryCard
                    title="Total Events"
                    value={summaryData.total_records.toString()}
                    icon="📊"
                  />
                  <SummaryCard
                    title="Duration"
                    value={summaryData.duration_ms ? formatDuration(summaryData.duration_ms) : 'N/A'}
                    icon="⏱️"
                  />
                  <SummaryCard
                    title="Event Types"
                    value={summaryData.event_types.length.toString()}
                    icon="🏷️"
                  />
                  <SummaryCard
                    title="Status"
                    value={summaryData.event_types.includes('COMPLETED') ? 'Completed' : summaryData.event_types.includes('FAILED') ? 'Failed' : 'In Progress'}
                    icon={summaryData.event_types.includes('COMPLETED') ? '✅' : summaryData.event_types.includes('FAILED') ? '❌' : '🔄'}
                  />
                </div>

                {/* Timeline Summary */}
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Timeline Summary</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">First Event:</span>
                      <span className="font-medium">
                        {summaryData.first_event
                          ? formatTimestamp(summaryData.first_event, {
                              dateStyle: 'medium',
                              timeStyle: 'short',
                            })
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Last Event:</span>
                      <span className="font-medium">
                        {summaryData.last_event
                          ? formatTimestamp(summaryData.last_event, {
                              dateStyle: 'medium',
                              timeStyle: 'short',
                            })
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Total Duration:</span>
                      <span className="font-medium">
                        {summaryData.duration_ms !== null
                          ? formatDuration(summaryData.duration_ms)
                          : 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Event Types */}
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Event Types</h3>
                  <div className="flex flex-wrap gap-2">
                    {summaryData.event_types.map((eventType) => (
                      <span
                        key={eventType}
                        className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium"
                      >
                        {eventType}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center text-gray-500">
                <p>No summary data available.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// Summary Card Component
const SummaryCard: React.FC<{
  title: string;
  value: string;
  icon: string;
}> = ({ title, value, icon }) => (
  <div className="bg-white border border-gray-200 rounded-lg p-4">
    <div className="flex items-center justify-between mb-2">
      <span className="text-sm text-gray-600">{title}</span>
      <span className="text-2xl">{icon}</span>
    </div>
    <div className="text-2xl font-bold text-gray-900">{value}</div>
  </div>
);

export default ProvenanceViewer;
