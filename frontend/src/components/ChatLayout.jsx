import React from 'react';
import Header from './Header';

const ChatLayout = ({ children, onLogout, onClearChat, username }) => {
    return (
        <div className="flex flex-col h-[100dvh] bg-[#fdfdfe] selection:bg-blue-100">
            <Header onLogout={onLogout} onClearChat={onClearChat} username={username} />

            <main className="flex-1 overflow-hidden flex flex-col relative">
                {/* Message area with custom scrollbar */}
                <div className="flex-1 overflow-y-auto px-4 md:px-0 py-6 scroll-smooth custom-scrollbar">
                    <div className="max-w-4xl mx-auto flex flex-col gap-1">
                        {children}
                    </div>
                </div>
            </main>
        </div>
    );
};

export default ChatLayout;
