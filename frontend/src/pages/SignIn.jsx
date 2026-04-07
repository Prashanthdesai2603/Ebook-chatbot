import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Eye, EyeOff, Lock, User, Mail, Loader2, Check, X, ShieldCheck } from 'lucide-react';

const SignIn = () => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  // Password requirements state
  const [requirements, setRequirements] = useState({
    length: false,
    upper: false,
    lower: false,
    number: false,
    special: false
  });

  const [strength, setStrength] = useState({ label: 'Weak', color: 'bg-gray-200', width: '0%' });

  useEffect(() => {
    const pass = formData.password;
    const reqs = {
      length: pass.length >= 8,
      upper: /[A-Z]/.test(pass),
      lower: /[a-z]/.test(pass),
      number: /[0-9]/.test(pass),
      special: /[@#$%^&+=!]/.test(pass)
    };
    setRequirements(reqs);

    // Calculate strength
    const metCount = Object.values(reqs).filter(Boolean).length;
    if (metCount === 0) setStrength({ label: 'None', color: 'bg-gray-200', width: '0%' });
    else if (metCount <= 2) setStrength({ label: 'Weak', color: 'bg-red-500', width: '33%' });
    else if (metCount <= 4) setStrength({ label: 'Medium', color: 'bg-yellow-500', width: '66%' });
    else setStrength({ label: 'Strong', color: 'bg-emerald-500', width: '100%' });
  }, [formData.password]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const validateEmail = (email) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  };

  const handleSignIn = async (e) => {
    e.preventDefault();
    setError('');

    if (!validateEmail(formData.email)) {
      setError('Invalid email format');
      return;
    }

    if (Object.values(requirements).some(v => !v)) {
      setError('Password does not meet all requirements');
      return;
    }

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/api/signup', {
        username: formData.username,
        email: formData.email,
        password: formData.password
      });

      if (response.data.success) {
        navigate('/login', { state: { message: 'Account created successfully! Please log in.' } });
      }
    } catch (err) {
      const errMsg = err.response?.data?.detail || 'Failed to create account. Please try again.';
      setError(errMsg);
    } finally {
      setIsLoading(false);
    }
  };

  const RequirementItem = ({ met, text }) => (
    <li className={`flex items-center gap-1.5 ${met ? 'text-emerald-600' : 'text-gray-400'} transition-colors duration-200`}>
      {met ? <Check size={12} className="stroke-[3]" /> : <X size={12} className="stroke-[3]" />}
      <span>{text}</span>
    </li>
  );

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
      <div className="max-w-md w-full bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight mb-1">
            Create Account
          </h1>
          <p className="text-gray-400 text-sm font-medium">Join the Injection Molding Assistant</p>
        </div>

        <form onSubmit={handleSignIn} className="space-y-4">
          {error && (
            <div className="bg-red-50 text-red-600 text-xs font-bold p-3 rounded-xl border border-red-100 animate-in fade-in slide-in-from-top-1">
              {error}
            </div>
          )}

          <div className="grid grid-cols-2 gap-4">
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
                  name="username"
                  value={formData.username}
                  onChange={handleInputChange}
                  className="block w-full pl-9 pr-3 py-2 border border-gray-200 rounded-xl text-sm bg-white placeholder-gray-300 focus:outline-none focus:ring-4 focus:ring-blue-500/10 focus:border-blue-500 transition-all shadow-sm"
                  placeholder="Enter your username"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-gray-700 mb-1.5 uppercase tracking-wider">
                Email
              </label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400 group-focus-within:text-blue-500 transition-colors">
                  <Mail size={16} />
                </div>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  className="block w-full pl-9 pr-3 py-2 border border-gray-200 rounded-xl text-sm bg-white placeholder-gray-300 focus:outline-none focus:ring-4 focus:ring-blue-500/10 focus:border-blue-500 transition-all shadow-sm"
                  placeholder="example@gmail.com"
                  required
                />
              </div>
            </div>
          </div>

          <div>
            <label className="block text-xs font-bold text-gray-700 mb-1.5 uppercase tracking-wider uppercase tracking-wider">
              Password
            </label>
            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400 group-focus-within:text-blue-500 transition-colors">
                <Lock size={16} />
              </div>
              <input
                type={showPassword ? 'text' : 'password'}
                name="password"
                value={formData.password}
                onChange={handleInputChange}
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

          <div>
            <label className="block text-xs font-bold text-gray-700 mb-1.5 uppercase tracking-wider uppercase tracking-wider">
              Confirm Password
            </label>
            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400 group-focus-within:text-blue-500 transition-colors">
                <Lock size={16} />
              </div>
              <input
                type={showConfirmPassword ? 'text' : 'password'}
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleInputChange}
                className="block w-full pl-9 pr-20 py-2 border border-gray-200 rounded-xl text-sm bg-white placeholder-gray-300 focus:outline-none focus:ring-4 focus:ring-blue-500/10 focus:border-blue-500 transition-all shadow-sm"
                placeholder="••••••••"
                required
              />
              <button
                type="button"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center gap-1.5 text-gray-400 hover:text-blue-600 transition-colors cursor-pointer group/btn"
              >
                <span className="text-[10px] font-bold uppercase opacity-0 group-hover/btn:opacity-100 transition-opacity">
                  {showConfirmPassword ? 'Hide' : 'Show'}
                </span>
                {showConfirmPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {/* Strength Meter */}
          {formData.password && (
            <div className="space-y-1.5 animate-in fade-in zoom-in-95 duration-300">
              <div className="flex justify-between items-center px-1">
                <span className="text-[10px] font-bold text-gray-400 uppercase tracking-tighter">Strength: {strength.label}</span>
                <span className="text-[10px] font-bold text-gray-400 uppercase tracking-tighter">{strength.width}</span>
              </div>
              <div className="h-1 w-full bg-gray-100 rounded-full overflow-hidden">
                <div 
                  className={`h-full ${strength.color} transition-all duration-500 ease-out`} 
                  style={{ width: strength.width }}
                />
              </div>
              
              <ul className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2 px-1">
                <RequirementItem met={requirements.length} text="8+ Characters" />
                <RequirementItem met={requirements.upper} text="1 Uppercase" />
                <RequirementItem met={requirements.lower} text="1 Lowercase" />
                <RequirementItem met={requirements.number} text="1 Number" />
                <RequirementItem met={requirements.special} text="1 Special char" />
              </ul>
            </div>
          )}

          <div className="pt-2">
            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex justify-center items-center py-2.5 px-4 border border-transparent rounded-xl shadow-lg text-sm font-bold text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all active:scale-[0.98]"
            >
              {isLoading ? (
                <>
                  <Loader2 className="animate-spin mr-2" size={18} />
                  Processing...
                </>
              ) : (
                'Complete Sign In'
              )}
            </button>
          </div>
        </form>

        <div className="mt-4 flex items-center justify-center gap-2 text-emerald-600 bg-emerald-50 py-2 rounded-xl border border-emerald-100">
          <ShieldCheck size={14} />
          <span className="text-[10px] font-bold uppercase tracking-wider">Your password is securely encrypted</span>
        </div>

        <div className="mt-6 text-center">
          <p className="text-xs text-gray-500">
            Already have an account?{' '}
            <button
              onClick={() => navigate('/login')}
              className="text-blue-600 font-bold hover:text-blue-700 transition-colors cursor-pointer"
            >
              Log In
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

export default SignIn;
