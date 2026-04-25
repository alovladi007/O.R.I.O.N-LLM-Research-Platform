// @ts-nocheck — MUI sx-prop union (project pattern).
'use client'

/**
 * Phase 9 / Session 9.3 — virtualized log tail.
 *
 * Tails ``GET /jobs/{id}/logs`` (currently a one-shot text body
 * placeholder; Session 1.4 noted real streaming lands in 2.1) AND
 * subscribes to the SSE event stream so terminal transitions
 * surface immediately. Buffer cap: 10 000 lines (older falls off the
 * top with a "view full log" hint to the artifact tab).
 *
 * Auto-scroll pause-on-scroll: if the user scrolls up, we stop
 * pinning to the bottom; a "Resume tailing" badge shows. Resumes
 * automatically when they scroll back to the bottom.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Box,
  Button,
  Chip,
  IconButton,
  Stack,
  Typography,
  Tooltip,
} from '@mui/material'
import { Download, Pause, PlayArrow } from '@mui/icons-material'
import { FixedSizeList as VirtualList } from 'react-window'
import { saveAs } from 'file-saver'

import { api, openSse, type SseHandle } from '@/lib/api'

const MAX_BUFFER_LINES = 10_000
const ROW_HEIGHT = 18

interface Props {
  jobId: string
  height?: number
}

export function LogTail({ jobId, height = 480 }: Props) {
  const [lines, setLines] = useState<string[]>([])
  const [autoScroll, setAutoScroll] = useState(true)
  const [terminal, setTerminal] = useState(false)
  const listRef = useRef<VirtualList | null>(null)

  // Pull initial tail.
  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const text = await api.jobs.logsText(jobId, 500)
        if (!cancelled) setLines(text.split('\n'))
      } catch {
        /* ignore */
      }
    })()
    return () => {
      cancelled = true
    }
  }, [jobId])

  // Subscribe to SSE; on terminal, refetch the full one-shot log so
  // the user sees the trailing lines that the polling SSE didn't
  // emit as separate events.
  useEffect(() => {
    let handle: SseHandle | null = null
    handle = openSse(`/jobs/${jobId}/events`, {
      on: {
        terminal: () => {
          setTerminal(true)
          // One last full fetch.
          void (async () => {
            try {
              const text = await api.jobs.logsText(jobId, 1000)
              setLines(text.split('\n'))
            } catch {
              /* ignore */
            }
          })()
        },
        status: (d: any) => {
          if (d.current_step) {
            setLines((prev) =>
              cap(prev.concat([`[${d.status}] step=${d.current_step}`])),
            )
          }
        },
      },
    })
    return () => handle?.cancel()
  }, [jobId])

  // Auto-scroll to bottom when new lines arrive (unless paused).
  useEffect(() => {
    if (autoScroll && listRef.current) {
      listRef.current.scrollToItem(lines.length - 1, 'end')
    }
  }, [lines, autoScroll])

  const downloadLog = useCallback(() => {
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' })
    saveAs(blob, `${jobId}.log`)
  }, [lines, jobId])

  return (
    <Box>
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
        <Typography variant="caption" color="text.secondary">
          {lines.length} lines
        </Typography>
        {terminal && <Chip label="TERMINAL" color="warning" size="small" />}
        <Box sx={{ flexGrow: 1 }} />
        <Tooltip title={autoScroll ? 'Pause auto-scroll' : 'Resume auto-scroll'}>
          <IconButton
            size="small"
            onClick={() => setAutoScroll((s) => !s)}
            data-testid="log-toggle-autoscroll"
          >
            {autoScroll ? <Pause fontSize="small" /> : <PlayArrow fontSize="small" />}
          </IconButton>
        </Tooltip>
        <Tooltip title="Download as .log">
          <IconButton
            size="small"
            onClick={downloadLog}
            data-testid="log-download"
          >
            <Download fontSize="small" />
          </IconButton>
        </Tooltip>
      </Stack>
      <Box
        sx={{
          bgcolor: '#0a0a14',
          color: '#cccccc',
          fontFamily: 'monospace',
          fontSize: 13,
          borderRadius: 1,
          overflow: 'hidden',
        }}
        data-testid="log-tail"
      >
        <VirtualList
          ref={listRef}
          height={height}
          itemCount={lines.length}
          itemSize={ROW_HEIGHT}
          width="100%"
          onScroll={({ scrollOffset, scrollUpdateWasRequested }) => {
            if (scrollUpdateWasRequested) return
            const nearBottom =
              scrollOffset + height >= lines.length * ROW_HEIGHT - ROW_HEIGHT
            setAutoScroll(nearBottom)
          }}
        >
          {({ index, style }) => (
            <div style={{ ...style, padding: '0 12px', whiteSpace: 'pre' }}>
              {lines[index]}
            </div>
          )}
        </VirtualList>
      </Box>
      {lines.length >= MAX_BUFFER_LINES && (
        <Typography variant="caption" color="warning.main" sx={{ mt: 0.5, display: 'block' }}>
          Buffer cap reached ({MAX_BUFFER_LINES} lines). Older lines were
          dropped — see the Artifacts tab for the full log.
        </Typography>
      )}
    </Box>
  )
}

function cap(lines: string[]): string[] {
  if (lines.length <= MAX_BUFFER_LINES) return lines
  return lines.slice(lines.length - MAX_BUFFER_LINES)
}
