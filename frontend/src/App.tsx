import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useEffect } from 'react';
import { Hexagon, Loader2 } from 'lucide-react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ProtectedRoute } from './components/auth/ProtectedRoute';
import { PublicOnlyRoute } from './components/auth/PublicOnlyRoute';
import { Sidebar } from './components/layout/Sidebar';
import { ProjectsHubView } from './pages/ProjectsHubView';
import { ProjectHomeView } from './pages/ProjectHomeView';
import { ConversationView } from './pages/ConversationView';
import { LoginView } from './pages/LoginView';
import { SignupView } from './pages/SignupView';
import { checkHealth } from './services/api';

function AppRoutes() {
  const { isAuthenticated, isLoading } = useAuth();

  useEffect(() => {
    // Basic backend connectivity check
    checkHealth().then((isHealthy) => {
      console.log(`Backend connectivity test: ${isHealthy ? 'SUCCESS' : 'FAILED'}`);
    });
  }, []);

  if (isLoading) {
    return (
      <div className="fixed inset-0 flex flex-col items-center justify-center bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-zinc-900 via-zinc-950 to-black text-[#FAFAFA] font-sans">
        <div className="w-11 h-11 rounded-xl bg-indigo-500/20 border border-indigo-500/30 shadow-[0_0_24px_rgba(99,102,241,0.4)] flex items-center justify-center mb-4 animate-pulse">
          <Hexagon className="w-6 h-6 text-indigo-400" />
        </div>
        <div className="flex items-center gap-2.5 text-zinc-400 text-sm font-medium">
          <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
          <span>Initializing workspace...</span>
        </div>
      </div>
    );
  }

  return (
    <Routes>
      {/* Root redirect: if authenticated -> /projects, if unauthenticated -> /login */}
      <Route
        path="/"
        element={
          isAuthenticated ? (
            <Navigate to="/projects" replace />
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />

      {/* Public Only Auth Routes */}
      <Route
        path="/login"
        element={
          <PublicOnlyRoute>
            <LoginView />
          </PublicOnlyRoute>
        }
      />
      <Route
        path="/signup"
        element={
          <PublicOnlyRoute>
            <SignupView />
          </PublicOnlyRoute>
        }
      />

      {/* Protected Application Routes wrapped in Sidebar Layout */}
      <Route
        path="/*"
        element={
          <ProtectedRoute>
            <Sidebar>
              <Routes>
                {/* Projects Hub */}
                <Route path="/projects" element={<ProjectsHubView />} />

                {/* Project Home (Tabs: chats | sources) */}
                <Route path="/projects/:projectId" element={<ProjectHomeView />} />

                {/* Conversation View */}
                <Route path="/projects/:projectId/c/:conversationId" element={<ConversationView />} />

                {/* Fallback inside workspace */}
                <Route path="*" element={<Navigate to="/projects" replace />} />
              </Routes>
            </Sidebar>
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <div className="fixed inset-0 overflow-hidden bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-zinc-900 via-zinc-950 to-black text-[#FAFAFA] font-sans">
          <AppRoutes />
        </div>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
