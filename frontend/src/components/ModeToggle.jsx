import React from 'react';

const ModeToggle = ({ mode, setMode }) => {
    return (
        <div className="flex bg-gray-100/80 p-1.5 rounded-[100px] w-fit shadow-inner ring-1 ring-black/5 backdrop-blur-sm">
            <button
                className={`
                    px-6 py-2.5 text-[11px] font-bold uppercase tracking-wider rounded-[100px] transition-all duration-300
                    ${mode === 'short'
                        ? 'bg-blue-600 text-white shadow-md shadow-blue-200'
                        : 'text-gray-500 hover:text-gray-700'
                    }
                `}
                onClick={() => setMode('short')}
            >
                Short
            </button>
            <button
                className={`
                    px-6 py-2.5 text-[11px] font-bold uppercase tracking-wider rounded-[100px] transition-all duration-300
                    ${mode === 'detailed'
                        ? 'bg-blue-600 text-white shadow-md shadow-blue-200'
                        : 'text-gray-500 hover:text-gray-700'
                    }
                `}
                onClick={() => setMode('detailed')}
            >
                Detailed
            </button>
        </div>
    );
};

export default ModeToggle;
