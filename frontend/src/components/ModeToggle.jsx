import React from 'react';

const ModeToggle = ({ mode, setMode }) => {
    return (
        <div className="flex bg-gray-100 p-1 rounded-xl mb-3 w-fit mx-auto">
            <button
                className={`px-4 py-1.5 text-xs font-semibold rounded-lg transition-all ${mode === 'short'
                    ? 'bg-white text-primary shadow-sm'
                    : 'text-gray-500 hover:text-gray-700'
                    }`}
                onClick={() => setMode('short')}
            >
                Short Mode
            </button>
            <button
                className={`px-4 py-1.5 text-xs font-semibold rounded-lg transition-all ${mode === 'detailed'
                    ? 'bg-white text-primary shadow-sm'
                    : 'text-gray-500 hover:text-gray-700'
                    }`}
                onClick={() => setMode('detailed')}
            >
                Detailed Mode
            </button>
        </div>
    );
};

export default ModeToggle;
