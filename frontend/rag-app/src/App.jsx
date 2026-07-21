import { useState } from "react";
import ChatBox from "./components/ChatBox";
import UploadPDF from "./components/UploadPDF";
import DocumentList from "./components/DocumentList";

function App() {
  const [messages, setMessages] = useState([]);
  const [refreshKey, setRefreshKey] = useState(0);

  return (
    <div className="app">
      <header className="app-header">
        <h1>RAG Chat</h1>
        <p className="app-subtitle">Ask questions grounded in your uploaded PDFs</p>
      </header>

      <main className="layout">
        <section className="chat-container">
          <div className="chat-toolbar">
            <span className="chat-toolbar-label">Conversation</span>
            <button
              className="clear-btn"
              onClick={() => setMessages([])}
              disabled={messages.length === 0}
            >
              Clear
            </button>
          </div>

          <ChatBox messages={messages} setMessages={setMessages} />
        </section>

        <aside className="sidebar">
          <h2>Documents</h2>
          <UploadPDF onUploaded={() => setRefreshKey((k) => k + 1)} />
          <DocumentList refreshKey={refreshKey} />
        </aside>
      </main>
    </div>
  );
}

export default App;