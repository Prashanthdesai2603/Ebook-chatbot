import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import { Eye, EyeOff, Lock, User, Loader2, CheckCircle2 } from 'lucide-react';

const Login = () => {
  const location = useLocation();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState(location.state?.message || '');
  const navigate = useNavigate();

  useEffect(() => {
    if (username || password) {
      setSuccessMessage('');
    }
  }, [username, password]);

  useEffect(() => {
    const token = localStorage.getItem('token');
    const isAuthenticated = token && token !== 'undefined' && token !== 'null';
    if (isAuthenticated) {
      navigate('/chat');
    }
  }, [navigate]);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${import.meta.env.VITE_API_URL}/login`, {
        username,
        password,
      });

      if (response.data.success) {
        localStorage.setItem('token', response.data.token);
        localStorage.setItem('username', username);
        navigate('/chat');
      }
    } catch (err) {
      setError('Invalid username or password');
      console.error('Login error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
      <div className="max-w-md w-full bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight mb-1">
            Injection Molding Assistant
          </h1>
          <p className="text-gray-400 text-sm font-medium">Login to continue</p>
        </div>

        <form onSubmit={handleLogin} className="space-y-4">
          {successMessage && (
            <div className="bg-emerald-50 text-emerald-600 text-xs font-bold p-3 rounded-xl border border-emerald-100 flex items-center gap-2 animate-in fade-in slide-in-from-top-1">
              <CheckCircle2 size={14} />
              {successMessage}
            </div>
          )}

          {error && (
            <div className="bg-red-50 text-red-600 text-xs font-bold p-3 rounded-xl border border-red-100 animate-pulse">
              {error}
            </div>
          )}

          <div>
            <label className="block text-xs font-bold text-gray-700 mb-1.5 uppercase tracking-wider">
              Username
            </label>
            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400 group-focus-within:text-blue-500 transition-colors">
                <User size={16} />
              </div>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="block w-full pl-9 pr-3 py-2 border border-gray-200 rounded-xl text-sm bg-white placeholder-gray-300 focus:outline-none focus:ring-4 focus:ring-blue-500/10 focus:border-blue-500 transition-all shadow-sm"
                placeholder="Enter your username"
                required
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-bold text-gray-700 mb-1.5 uppercase tracking-wider">
              Password
            </label>
            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400 group-focus-within:text-blue-500 transition-colors">
                <Lock size={16} />
              </div>
              <input
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="block w-full pl-9 pr-20 py-2 border border-gray-200 rounded-xl text-sm bg-white placeholder-gray-300 focus:outline-none focus:ring-4 focus:ring-blue-500/10 focus:border-blue-500 transition-all shadow-sm"
                placeholder="••••••••"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center gap-1.5 text-gray-400 hover:text-blue-600 transition-colors cursor-pointer group/btn"
              >
                <span className="text-[10px] font-bold uppercase opacity-0 group-hover/btn:opacity-100 transition-opacity">
                  {showPassword ? 'Hide' : 'Show'}
                </span>
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <div className="pt-2">
            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex justify-center items-center py-2.5 px-4 border border-transparent rounded-xl shadow-lg text-sm font-bold text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all active:scale-[0.98]"
            >
              {isLoading ? (
                <>
                  <Loader2 className="animate-spin mr-2" size={18} />
                  Authenticating...
                </>
              ) : (
                'Login'
              )}
            </button>
          </div>
        </form>

        <div className="mt-4">
          <button
            onClick={() => navigate('/signin')}
            className="w-full flex justify-center items-center py-2.5 px-4 border border-gray-200 rounded-xl shadow-sm text-sm font-bold text-gray-400 bg-white hover:bg-gray-50 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all cursor-pointer"
          >
            Create New Account
          </button>
        </div>

        <div className="mt-6 pt-4 border-t border-gray-100 text-center">
          <p className="text-[10px] text-gray-400 uppercase tracking-widest font-bold">
            SECURE ACCESS
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
