import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { AlertCircle, CheckCircle2 } from 'lucide-react';

export const SignupView: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  
  const [emailError, setEmailError] = useState('');
  const [passwordError, setPasswordError] = useState('');
  
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const validateEmail = (val: string) => {
    if (!val.trim()) return 'Email is required';
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(val)) return 'Please enter a valid email address';
    return '';
  };

  const validatePassword = (val: string) => {
    if (!val) return 'Password is required';
    if (val.length < 8) return 'Password must be at least 8 characters';
    return '';
  };

  const handleSignup = (e: React.FormEvent) => {
    e.preventDefault();
    setHasSubmitted(true);
    
    const eError = validateEmail(email);
    const pError = validatePassword(password);
    
    setEmailError(eError);
    setPasswordError(pError);
    
    if (!eError && !pError) {
      setIsSuccess(true);
    } else {
      setIsSuccess(false);
    }
  };

  const handleEmailChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEmail(e.target.value);
    if (hasSubmitted) setEmailError(validateEmail(e.target.value));
  };

  const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPassword(e.target.value);
    if (hasSubmitted) setPasswordError(validatePassword(e.target.value));
  };

  const getInputStyle = (hasError: boolean): React.CSSProperties => ({
    width: '100%',
    padding: '10px 14px',
    borderRadius: '8px',
    border: `1px solid ${hasError ? 'rgba(239,68,68,0.5)' : '#2a2a2a'}`,
    background: hasError ? 'rgba(239,68,68,0.02)' : '#141414',
    color: '#FFFFFF',
    fontSize: '14px',
    fontFamily: 'Inter, sans-serif',
    outline: 'none',
    transition: 'border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease',
    boxSizing: 'border-box',
  });

  const handleFocus = (e: React.FocusEvent<HTMLInputElement>, hasError: boolean) => {
    e.target.style.borderColor = hasError ? '#ef4444' : 'rgba(99,102,241,0.5)';
    e.target.style.background = hasError ? 'rgba(239,68,68,0.05)' : '#1a1a1a';
    e.target.style.boxShadow = hasError 
      ? '0 0 0 3px rgba(239,68,68,0.1)'
      : '0 0 0 3px rgba(99,102,241,0.1)';
  };
  
  const handleBlur = (e: React.FocusEvent<HTMLInputElement>, hasError: boolean) => {
    e.target.style.borderColor = hasError ? 'rgba(239,68,68,0.5)' : '#2a2a2a';
    e.target.style.background = hasError ? 'rgba(239,68,68,0.02)' : '#141414';
    e.target.style.boxShadow = 'none';
    
    // Validate on blur if not submitted yet
    if (!hasSubmitted) {
      if (e.target.id === 'email') setEmailError(validateEmail(email));
      if (e.target.id === 'password') setPasswordError(validatePassword(password));
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-zinc-900 via-zinc-950 to-black text-[#FAFAFA] font-sans p-4">
      <div 
        className="w-full max-w-[400px] p-8"
        style={{
          background: '#16181E',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: '16px',
          boxShadow: '0 24px 64px rgba(0,0,0,0.5)',
        }}
      >
        <div className="text-center mb-8">
          <h1 className="text-2xl font-semibold tracking-tight text-white mb-2">Create your account</h1>
          <p className="text-[14px] text-zinc-400">Sign up to get started with your workspace</p>
        </div>

        {isSuccess && (
          <div className="mb-6 p-4 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-start gap-3">
            <CheckCircle2 className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
            <div>
              <p className="text-[13.5px] font-medium text-indigo-300 mb-1">Validation passed</p>
              <p className="text-[12.5px] text-indigo-400/80">Client-side validation successful. Backend integration pending for next phase.</p>
            </div>
          </div>
        )}

        <form onSubmit={handleSignup} className="space-y-5" noValidate>
          <div>
            <label htmlFor="email" className="block text-[12px] font-semibold tracking-wide text-zinc-400 uppercase mb-2">
              Email
            </label>
            <input
              id="email"
              name="email"
              type="email"
              value={email}
              onChange={handleEmailChange}
              onFocus={(e) => handleFocus(e, !!emailError)}
              onBlur={(e) => handleBlur(e, !!emailError)}
              style={getInputStyle(!!emailError)}
              placeholder="Enter your email"
              aria-invalid={!!emailError}
              aria-describedby={emailError ? "email-error" : undefined}
            />
            {emailError && (
              <div id="email-error" className="flex items-center gap-1.5 mt-2 text-red-400">
                <AlertCircle size={14} />
                <span className="text-[12.5px]">{emailError}</span>
              </div>
            )}
          </div>

          <div>
            <label htmlFor="password" className="block text-[12px] font-semibold tracking-wide text-zinc-400 uppercase mb-2">
              Password
            </label>
            <input
              id="password"
              name="password"
              type="password"
              value={password}
              onChange={handlePasswordChange}
              onFocus={(e) => handleFocus(e, !!passwordError)}
              onBlur={(e) => handleBlur(e, !!passwordError)}
              style={getInputStyle(!!passwordError)}
              placeholder="••••••••"
              aria-invalid={!!passwordError}
              aria-describedby={passwordError ? "password-error" : undefined}
            />
            {passwordError && (
              <div id="password-error" className="flex items-center gap-1.5 mt-2 text-red-400">
                <AlertCircle size={14} />
                <span className="text-[12.5px]">{passwordError}</span>
              </div>
            )}
          </div>

          <button
            type="submit"
            className="w-full mt-2 inline-flex items-center justify-center px-4 py-2.5 rounded-xl text-[14px] font-semibold tracking-tight transition-all duration-200 bg-gradient-to-tr from-indigo-600 to-indigo-500 text-white shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/50 hover:-translate-y-0.5 active:scale-[0.98]"
          >
            Sign Up
          </button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-[13.5px] text-zinc-400">
            Already have an account?{' '}
            <Link to="/login" className="text-indigo-400 hover:text-indigo-300 font-medium transition-colors">
              Login
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};
