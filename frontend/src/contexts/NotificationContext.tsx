'use client'

import { createContext, useContext, ReactNode } from 'react'

const NotificationContext = createContext({})

export function NotificationProvider({ children }: { children: ReactNode }) {
  return <NotificationContext.Provider value={{}}>{children}</NotificationContext.Provider>
}

export function useNotifications() {
  return useContext(NotificationContext)
}
