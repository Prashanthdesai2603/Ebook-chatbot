import React from 'react';

const ChatLayout = ({ children }) => {
    return (
        <div className="flex flex-col h-screen bg-[#f7f7f8]">
            <header className="bg-white/80 backdrop-blur-md border-b border-gray-100 py-4 px-6 sticky top-0 z-10">
                <div className="max-w-4xl mx-auto flex justify-between items-center">
                    <div>
                        <h1 className="text-xl font-extrabold tracking-tight text-gray-900 flex items-center gap-2">
                            <span className="text-2xl">🤖</span>
                            Injection Molding Assistant
                        </h1>
                        <p className="text-xs text-gray-400 font-medium">AI-powered knowledge assistant for plastics professionals</p>
                    </div>
                    <div className="hidden sm:flex items-center gap-2 px-3 py-1 bg-blue-50 text-blue-600 rounded-full text-[10px] font-bold uppercase tracking-widest border border-blue-100">
                        Offline Enabled
                    </div>
                </div>
            </header>
            <main className="flex-1 overflow-hidden relative">
                <div className="h-full overflow-y-auto pt-8 pb-32 px-4 scroll-smooth">
                    <div className="max-w-4xl mx-auto">
                        {children}
                    </div>
                </div>
            </main>
        </div>
    );
};

export default ChatLayout;
