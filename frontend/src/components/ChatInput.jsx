import React, { useRef, useEffect, useState, useCallback } from 'react';

const ChatInput = ({ input, setInput, onSend, loading, mode, setMode }) => {
    const textareaRef = useRef(null);
    const [isListening, setIsListening] = useState(false);
    const recognitionRef = useRef(null);

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
        }
    }, [input]);

    useEffect(() => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            recognitionRef.current = new SpeechRecognition();
            recognitionRef.current.continuous = true;
            recognitionRef.current.interimResults = true;
            recognitionRef.current.lang = 'en-US';

            recognitionRef.current.onresult = (event) => {
                const transcript = Array.from(event.results)
                    .map(result => result[0])
                    .map(result => result.transcript)
                    .join('');

                setInput(transcript);
            };

            recognitionRef.current.onerror = (event) => {
                console.error('Speech recognition error', event.error);
                setIsListening(false);
            };

            recognitionRef.current.onend = () => {
                setIsListening(false);
            };
        }
    }, [setInput]);

    const toggleListen = useCallback(() => {
        if (!recognitionRef.current) {
            alert('Speech Recognition is not supported in this browser.');
            return;
        }

        if (isListening) {
            recognitionRef.current.stop();
            setIsListening(false);
        } else {
            try {
                recognitionRef.current.start();
                setIsListening(true);
            } catch (err) {
                console.error('Start error:', err);
                recognitionRef.current.stop();
                setIsListening(false);
            }
        }
    }, [isListening]);

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!loading && input.trim()) {
                if (isListening) {
                    recognitionRef.current.stop();
                    setIsListening(false);
                }
                onSend();
            }
        }
    };

    return (
        <div className="border-t border-gray-100 bg-white/80 backdrop-blur-md p-4 sticky bottom-0">
            <div className="max-w-4xl mx-auto">
                <div className="flex bg-white border border-gray-200 rounded-2xl p-2 shadow-sm focus-within:ring-2 focus-within:ring-primary/20 focus-within:border-primary transition-all items-end gap-2">
                    <textarea
                        ref={textareaRef}
                        rows="1"
                        className="flex-1 max-h-32 bg-transparent border-none focus:ring-0 text-sm py-2 px-3 resize-none scroll-smooth"
                        placeholder="Ask a question about injection molding..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                    />
                    <button
                        type="button"
                        onClick={toggleListen}
                        className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all relative ${isListening
                            ? 'bg-red-50 text-red-500 shadow-inner'
                            : 'bg-gray-50 text-gray-500 hover:bg-gray-100'
                            }`}
                        title={isListening ? "Stop listening" : "Start voice input"}
                    >
                        {isListening && (
                            <span className="absolute inset-0 rounded-xl bg-red-500/20 animate-ping"></span>
                        )}
                        <svg className={`h-5 w-5 ${isListening ? 'animate-pulse' : ''}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                            <line x1="12" y1="19" x2="12" y2="23"></line>
                            <line x1="8" y1="23" x2="16" y2="23"></line>
                        </svg>
                    </button>

                    <button
                        onClick={() => {
                            if (isListening) {
                                recognitionRef.current.stop();
                                setIsListening(false);
                            }
                            onSend();
                        }}
                        disabled={loading || !input.trim()}
                        className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all ${loading || !input.trim()
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                            : 'bg-primary text-white hover:scale-105 active:scale-95 shadow-md shadow-primary/20'
                            }`}
                    >
                        {loading ? (
                            <svg className="animate-spin h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                        ) : (
                            <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="22" y1="2" x2="11" y2="13"></line>
                                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                            </svg>
                        )}
                    </button>
                </div>
                <div className="mt-3 text-center">
                    <p className="text-[11px] text-gray-400 flex items-center justify-center gap-1.5">
                        <span className="flex h-1.5 w-1.5 rounded-full bg-green-500"></span>
                        Answers generated from ebook knowledge base
                    </p>
                </div>
            </div>
        </div>
    );
};

export default ChatInput;
