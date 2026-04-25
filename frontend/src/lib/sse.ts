/**
 * Phase 9 / Session 9.3 — typed SSE client.
 *
 * Wraps the browser's EventSource with a tiny lifecycle:
 *   - reconnect with exponential backoff (1 s → 30 s cap)
 *   - per-event-name handler dispatch (matches the backend's
 *     ``event: status\ndata: {…}\n\n`` framing)
 *   - cancel() to fully close (used in component unmount)
 *
 * EventSource sends cookies cross-origin only when ``withCredentials``
 * is true; we set it so the same orion_access_token cookie that the
 * REST client uses also authenticates the SSE stream. Note that
 * EventSource does NOT honor the axios refresh-on-401 interceptor —
 * if the token expires mid-stream, the connection errors and the
 * caller should rely on the page's normal refresh path.
 */

const DEFAULT_BACKOFF_MS = 1_000
const MAX_BACKOFF_MS = 30_000

export interface SseHandlers {
  /** Catch-all for any event whose name is not in the dispatch map. */
  onMessage?: (event: string, data: unknown) => void
  /** Per-event-name handlers. */
  on?: Record<string, (data: unknown) => void>
  /** Fires after every reconnect attempt; ``error`` may be null on first. */
  onReconnect?: (attempt: number, error: Event | null) => void
  /** Fires when the EventSource is finally closed (cancel + terminal). */
  onClose?: () => void
}

export interface SseHandle {
  cancel: () => void
}

function resolveBaseUrl(): string {
  if (typeof process !== 'undefined') {
    const fromEnv = process.env.NEXT_PUBLIC_ORION_API_URL
    if (fromEnv) return fromEnv
  }
  return 'http://localhost:8002/api/v1'
}

/**
 * Open an SSE stream against ``path`` (relative to the API base URL).
 *
 * Example:
 *
 *   const handle = openSse('/jobs/abc123/events', {
 *     on: {
 *       status: (d) => updateRow(d),
 *       terminal: (d) => stopPolling(),
 *     },
 *   })
 *   useEffect(() => () => handle.cancel(), [handle])
 */
export function openSse(path: string, handlers: SseHandlers): SseHandle {
  const url = `${resolveBaseUrl()}${path}`
  let backoff = DEFAULT_BACKOFF_MS
  let attempt = 0
  let cancelled = false
  let es: EventSource | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null

  const open = () => {
    if (cancelled) return
    es = new EventSource(url, { withCredentials: true })
    attempt += 1
    handlers.onReconnect?.(attempt, null)

    // Default ``message`` event (backend emits named events; this is
    // the fallback if a handler isn't registered).
    es.addEventListener('message', (ev) => {
      try {
        const data = JSON.parse((ev as MessageEvent).data)
        handlers.onMessage?.('message', data)
      } catch {
        /* ignore non-JSON */
      }
    })

    // Register named-event listeners explicitly. EventSource needs
    // each name added separately.
    for (const [name, fn] of Object.entries(handlers.on ?? {})) {
      es.addEventListener(name, (ev) => {
        try {
          const data = JSON.parse((ev as MessageEvent).data)
          fn(data)
        } catch {
          fn((ev as MessageEvent).data)
        }
      })
    }

    es.addEventListener('open', () => {
      // Successful open resets the backoff window.
      backoff = DEFAULT_BACKOFF_MS
    })

    es.addEventListener('error', (ev) => {
      // EventSource fires error on disconnect; close + retry.
      es?.close()
      es = null
      if (cancelled) return
      handlers.onReconnect?.(attempt, ev)
      reconnectTimer = setTimeout(open, backoff)
      backoff = Math.min(MAX_BACKOFF_MS, backoff * 2)
    })
  }

  open()

  return {
    cancel() {
      cancelled = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      es?.close()
      handlers.onClose?.()
    },
  }
}
