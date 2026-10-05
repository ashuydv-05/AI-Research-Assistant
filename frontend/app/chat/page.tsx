import { ChatContainer } from '@/component/chat/ChatContainer';
import { AppHeader } from '@/component/layout/AppHeader';

export default function ChatPage() {
  return (
    <div className="flex h-screen flex-col bg-white">
      <AppHeader />
      <ChatContainer />
    </div>
  );
}

