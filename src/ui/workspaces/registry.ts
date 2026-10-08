// Maps each model task to the session panel wording and the component that
// fills the main area. The shell (App, SessionPanel) only talks to this
// registry, so a new kind of model (image, speech, embeddings...) is added by
// extending `ModelTask` and registering a workspace here.

import type { ComponentType } from 'react';
import type { ModelEntry, ModelTask, Session } from '../types';
import { ChatIcon } from '../common/icons';
import { ChatWorkspace } from './chat/ChatWorkspace';

export interface WorkspaceProps {
  model: ModelEntry;
  session: Session;
  onRename: (title: string) => void;
}

export interface WorkspaceDefinition {
  /** Wording for the session panel, e.g. "Chats" / "New chat". */
  sessionLabel: { singular: string; plural: string };
  SessionIcon: ComponentType;
  Workspace: ComponentType<WorkspaceProps>;
}

export const workspaces: Record<ModelTask, WorkspaceDefinition> = {
  chat: {
    sessionLabel: { singular: 'chat', plural: 'Chats' },
    SessionIcon: ChatIcon,
    Workspace: ChatWorkspace,
  },
};
