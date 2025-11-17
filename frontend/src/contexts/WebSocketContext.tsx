'use client'

import { createContext, useContext, ReactNode } from 'react'

const WebSocketContext = createContext({})

export function WebSocketProvider({ children }: { children: ReactNode }) {
  return <WebSocketContext.Provider value={{}}>{children}</WebSocketContext.Provider>
}

export function useWebSocket() {
  return useContext(WebSocketContext)
}
