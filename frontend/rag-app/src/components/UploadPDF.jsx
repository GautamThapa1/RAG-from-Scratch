import { useState } from "react";
import { uploadPDF } from "../services/api";

function UploadPDF({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);

  const handleUpload = async () => {
    if (!file) return; // handles empty upload

    setUploading(true);
    try {
      const data = await uploadPDF(file);

      alert("Uploaded: " + data.filename);
      setFile(null); // reset the file 
      onUploaded?.();
    } catch (error) {
      console.log(error);
    }
    setUploading(false);
  };

  return (
    <div>
      <input
        type="file"
        accept=".pdf"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button onClick={handleUpload} disabled={!file || uploading}>
        {uploading ? "Uploading…" : "Digest PDF"}
      </button>
    </div>
  );
}

export default UploadPDF;