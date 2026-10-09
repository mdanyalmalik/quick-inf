// How much the saved chats take up, measured as data (see src/store/data-size.ts).

import { dataBytes } from '../store/data-size';
import { dbGetAll } from '../store/db';

export async function measureChatBytes(): Promise<number> {
  const [sessions, messages] = await Promise.all([dbGetAll('sessions'), dbGetAll('messages')]);
  return dataBytes(sessions) + dataBytes(messages);
}
