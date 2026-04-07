import React from 'react';
import { Copy, Check, CheckCheck } from 'lucide-react';

const ChatMessage = ({ message }) => {
    const isBot = message.sender === 'bot';
    const [copied, setCopied] = React.useState(false);

    const handleCopy = () => {
        navigator.clipboard.writeText(message.text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const formatTimestamp = () => {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    return (
        <div className={`flex w-full mb-3 px-2 md:px-0 ${isBot ? 'justify-start' : 'justify-end'} message-fade-in`}>
            <div className={`relative flex flex-col max-w-[75%] md:max-w-[70%] group transition-all duration-300`}>
                <div className={`
                    px-4 py-3 shadow-sm transition-all duration-200
                    ${isBot
                        ? 'bg-gray-100/80 backdrop-blur-sm text-gray-800 rounded-2xl rounded-tl-sm border border-white/50 ring-1 ring-black/5 hover:bg-gray-100 hover:shadow-md'
                        : 'bg-gradient-to-br from-blue-600 to-indigo-700 text-white rounded-2xl rounded-tr-sm shadow-indigo-100/50 hover:shadow-lg hover:shadow-blue-200/50'
                    }
                `}>
                    <div className="flex flex-col gap-1">
                        <div className={`text-[15px] leading-relaxed whitespace-pre-wrap font-medium`}>
                            {message.text}
                        </div>
                        
                        <div className={`flex items-center justify-end gap-1.5 mt-1 opacity-60`}>
                            <span className="text-[10px] font-medium tracking-tight">
                                {formatTimestamp()}
                            </span>
                            {!isBot && (
                                <span className="flex items-center gap-0.5">
                                    <CheckCheck className="w-3 h-3 text-white/90" />
                                </span>
                            )}
                        </div>
                    </div>

                    {isBot && (
                        <button
                            onClick={handleCopy}
                            className="absolute -right-10 top-2 p-2 rounded-lg bg-white border border-gray-100 text-gray-400 hover:text-blue-600 hover:bg-blue-50 transition-all opacity-0 group-hover:opacity-100 shadow-sm"
                            title="Copy message"
                        >
                            {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ChatMessage;
