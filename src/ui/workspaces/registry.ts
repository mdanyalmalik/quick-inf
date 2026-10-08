// Maps each model task to the session panel wording and the component that
// fills the main area. The shell (App, SessionPanel) only talks to this
// registry, so a new kind of model (image, speech, embeddings...) is added by
// extending `ModelTask` and registering a workspace here.

import type { ComponentType } from 'react';
import type { ModelEntry, ModelTask, Session } from '../types';
import { ChatIcon } from '../common/icons';
import { ChatWorkspace } from './chat/ChatWorkspace';
import { forgetSession } from './chat/chat-generation';

export interface WorkspaceProps {
  model: ModelEntry;
  session: Session;
  onRename: (title: string) => void;
  /** On phone-sized screens, returns to the session list; workspaces show a back button at the start of their header. */
  onBack?: () => void;
}

export interface WorkspaceDefinition {
  /** Wording for the session panel, e.g. "Chats" / "New chat". */
  sessionLabel: { singular: string; plural: string };
  SessionIcon: ComponentType;
  Workspace: ComponentType<WorkspaceProps>;
  /** Called before a session is deleted, to stop work in it and drop its cached data. */
  onDeleteSession?: (sessionId: string) => void;
}

export const workspaces: Record<ModelTask, WorkspaceDefinition> = {
  chat: {
    sessionLabel: { singular: 'chat', plural: 'Chats' },
    SessionIcon: ChatIcon,
    Workspace: ChatWorkspace,
    onDeleteSession: forgetSession,
  },
};
