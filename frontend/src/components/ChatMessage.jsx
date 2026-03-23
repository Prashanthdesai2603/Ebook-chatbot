import React from 'react';

const ChatMessage = ({ message }) => {
    const isBot = message.sender === 'bot';

    return (
        <div className={`flex w-full mb-6 ${isBot ? 'justify-start' : 'justify-end'} message-fade-in`}>
            <div className={`max-w-[80%] px-4 py-3 rounded-2xl shadow-sm ${isBot
                    ? 'bg-white text-gray-800 rounded-tl-none border border-gray-100'
                    : 'bg-primary text-white rounded-tr-none'
                }`}>
                <div className={`text-[10px] font-bold uppercase tracking-wider mb-1 opacity-50 ${isBot ? 'text-gray-500' : 'text-blue-100'
                    }`}>
                    {isBot ? 'AI Assistant' : 'You'}
                </div>
                <div className="text-[15px] leading-relaxed whitespace-pre-wrap">
                    {message.text}
                </div>
            </div>
        </div>
    );
};

export default ChatMessage;
