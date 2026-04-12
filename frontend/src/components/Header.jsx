import { useState, useRef, useEffect } from 'react';
import { LogOut, User, MessageSquare } from 'lucide-react';

const Header = ({ onLogout, onClearChat, username = "Professional User" }) => {
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const dropdownRef = useRef(null);

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setDropdownOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <header className="bg-white/80 backdrop-blur-md border-b border-gray-100 py-3 px-6 sticky top-0 z-50">
            <div className="max-w-5xl mx-auto flex justify-between items-center w-full">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-200">
                        <MessageSquare className="text-white w-5 h-5" />
                    </div>
                    <div>
                        <h1 className="text-lg font-bold tracking-tight text-gray-900 leading-tight">
                            Injection Molding AI
                        </h1>
                        <p className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider">Expert Knowledge System</p>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    <button
                        onClick={onClearChat}
                        className="text-xs font-semibold text-gray-500 hover:text-red-500 transition-colors px-3 py-2 rounded-lg hover:bg-red-50"
                    >
                        Clear Chat
                    </button>

                    <div className="h-6 w-px bg-gray-200 mx-1"></div>

                    {/* Profile Dropdown */}
                    <div className="relative" ref={dropdownRef}>
                        <div
                            className="flex items-center gap-3 cursor-pointer p-1 rounded-full hover:bg-gray-50 transition-all"
                            onClick={() => setDropdownOpen(!dropdownOpen)}
                        >
                            <div className="text-right hidden sm:block">
                                <p className="text-xs font-bold text-gray-800">{username}</p>
                                <p className="text-[10px] text-green-500 font-medium">Online</p>
                            </div>
                            <div className="w-9 h-9 bg-gray-100 border border-gray-200 rounded-full flex items-center justify-center overflow-hidden">
                                <User className="text-gray-400 w-5 h-5" />
                            </div>
                        </div>

                        {dropdownOpen && (
                            <div className="absolute right-0 mt-2 w-48 bg-white rounded-2xl shadow-xl border border-gray-100 py-2 z-50">
                                <div className="px-4 py-2 border-b border-gray-50 mb-1">
                                    <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Account</p>
                                </div>
                                <button
                                    onClick={() => { setDropdownOpen(false); onLogout(); }}
                                    className="w-full flex items-center gap-3 px-4 py-2.5 text-sm font-medium text-gray-600 hover:bg-red-50 hover:text-red-600 transition-colors"
                                >
                                    <LogOut className="w-4 h-4" />
                                    Logout
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </header>
    );
};

export default Header;
