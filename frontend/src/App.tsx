import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Sidebar } from './components/layout/Sidebar';
import { ProjectsHubView } from './pages/ProjectsHubView';
import { ProjectHomeView } from './pages/ProjectHomeView';
import { ConversationView } from './pages/ConversationView';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-[#0A0A0A] text-[#FAFAFA] font-sans">
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
