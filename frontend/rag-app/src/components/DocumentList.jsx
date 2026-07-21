import { useEffect, useState } from "react";
import { getDocuments, deleteDocument, deleteAllDocuments } from "../services/api";

function DocumentList({ refreshKey }) {
  const [documents, setDocuments] = useState([]);
  const [selected, setSelected] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const data = await getDocuments();
      setDocuments(data);
    } catch (error) {
      console.log(error);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchDocuments();
  }, [refreshKey]);

  const toggleSelect = (id) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((docId) => docId !== id) : [...prev, id]
    );
  };

  const handleDeleteSelected = async () => {
    for (const id of selected) {
      await deleteDocument(id);
    }
    setSelected([]);
    fetchDocuments();
  };

  const handleDeleteAll = async () => {
    await deleteAllDocuments();
    setSelected([]);
    fetchDocuments();
  };

  return (
    <div className="doc-list">
      <div className="doc-list-header">
        <span>Documents ({documents.length})</span>
        <button onClick={fetchDocuments} className="refresh-btn" disabled={loading}>
          {loading ? "…" : "Refresh"}
        </button>
      </div>

      {documents.length === 0 && (
        <p className="doc-list-empty">No documents uploaded yet.</p>
      )}

      <ul className="doc-list-items">
        {documents.map((doc) => (
          <li key={doc.id} className="doc-list-item">
            <label>
              <input
                type="checkbox"
                checked={selected.includes(doc.id)}
                onChange={() => toggleSelect(doc.id)}
              />
              <span className="doc-name">{doc.filename}</span>
              <span className="doc-meta">{doc.chunks} chunks</span>
            </label>
            <button
              className="doc-delete-btn"
              onClick={() => deleteDocument(doc.id).then(fetchDocuments)}
            >
              ✕
            </button>
          </li>
        ))}
      </ul>

      <div className="doc-list-actions">
        <button
          onClick={handleDeleteSelected}
          disabled={selected.length === 0}
          className="delete-selected-btn"
        >
          Delete selected ({selected.length})
        </button>
        <button
          onClick={handleDeleteAll}
          disabled={documents.length === 0}
          className="delete-all-btn"
        >
          Delete all
        </button>
      </div>
    </div>
  );
}

export default DocumentList;