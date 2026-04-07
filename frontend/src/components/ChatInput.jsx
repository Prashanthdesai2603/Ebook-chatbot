import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Send, Mic, MicOff, Loader2 } from 'lucide-react';
import ModeToggle from './ModeToggle';

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
        <div className="bg-gradient-to-t from-white via-white to-white/0 pt-10 pb-6 px-4 md:px-6">
            <div className="max-w-4xl mx-auto flex flex-col items-center gap-4">
                
                {/* Mode Toggle as Pill */}
                <ModeToggle mode={mode} setMode={setMode} />

                {/* Input Container */}
                <div className={`
                    w-full flex items-end gap-2 p-2.5 bg-white border border-gray-200 rounded-[24px] shadow-lg shadow-gray-200/50 
                    focus-within:border-blue-400 focus-within:ring-4 focus-within:ring-blue-50 transition-all duration-300
                `}>
                    <textarea
                        ref={textareaRef}
                        rows="1"
                        className="flex-1 max-h-48 bg-transparent border-none focus:ring-0 text-[15px] font-medium py-3 px-4 resize-none scroll-smooth placeholder:text-gray-400"
                        placeholder="Type your question here..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                    />
                    
                    <div className="flex items-center gap-1.5 pb-1 pr-1">
                        <button
                            type="button"
                            onClick={toggleListen}
                            className={`
                                w-11 h-11 rounded-full flex items-center justify-center transition-all relative
                                ${isListening
                                    ? 'bg-red-50 text-red-500 ring-2 ring-red-100'
                                    : 'bg-gray-50 text-gray-400 hover:bg-gray-100 hover:text-gray-600'
                                }
                            `}
                            title={isListening ? "Stop listening" : "Start voice input"}
                        >
                            {isListening && (
                                <span className="absolute inset-0 rounded-full bg-red-500/20 animate-ping"></span>
                            )}
                            {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
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
                            className={`
                                w-11 h-11 rounded-full flex items-center justify-center transition-all
                                ${loading || !input.trim()
                                    ? 'bg-gray-100 text-gray-300 cursor-not-allowed'
                                    : 'bg-blue-600 text-white hover:bg-blue-700 hover:scale-105 active:scale-95 shadow-lg shadow-blue-200'
                                }
                            `}
                        >
                            {loading ? (
                                <Loader2 className="w-5 h-5 animate-spin" />
                            ) : (
                                <Send className="w-5 h-5" />
                            )}
                        </button>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse"></div>
                    <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest text-center">
                        Verified Knowledge Base Connected
                    </p>
                </div>
            </div>
        </div>
    );
};

export default ChatInput;
