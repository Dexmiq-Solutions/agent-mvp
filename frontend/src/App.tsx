import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useEffect } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { ProjectsHubView } from './pages/ProjectsHubView';
import { ProjectHomeView } from './pages/ProjectHomeView';
import { ConversationView } from './pages/ConversationView';
import { checkHealth } from './services/api';

function App() {
  useEffect(() => {
    // Phase 1: Basic backend connectivity test on load
    checkHealth().then((isHealthy) => {
      console.log(`Backend connectivity test: ${isHealthy ? 'SUCCESS' : 'FAILED'}`);
    });
  }, []);

  return (
    <BrowserRouter>
      <div className="fixed inset-0 overflow-hidden bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-zinc-900 via-zinc-950 to-black text-[#FAFAFA] font-sans">
        <Routes>
          {/* Redirect root to projects */}
          <Route path="/" element={<Navigate to="/projects" replace />} />

          {/* Main App Routes wrapped in Sidebar Layout */}
          <Route
            path="/*"
            element={
              <Sidebar>
                <Routes>
                  {/* Projects Hub */}
                  <Route path="/projects" element={<ProjectsHubView />} />

                  {/* Project Home (Tabs: chats | sources) */}
                  <Route path="/projects/:projectId" element={<ProjectHomeView />} />

                  {/* Conversation View */}
                  <Route path="/projects/:projectId/c/:conversationId" element={<ConversationView />} />
                </Routes>
              </Sidebar>
            }
          />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
