import axios from "axios";


// custom instance to reduce writing the whole url for 
// every request
const API = axios.create({
  baseURL: "http://localhost:8000",
});

export const askQuestion = async (question) => {
  const response = await API.post("/ask", {
    question: question,
    top_k: 5,
    top_n: 3,
  });

  return response.data;
};

export const uploadPDF = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await API.post("/upload", formData);

  return response.data;
};

export const getDocuments = async () => {
  const response = await API.get("/documents");
  return response.data;
};

export const deleteDocument = async (id) => {
  const response = await API.delete(`/documents/${id}`);
  return response.data;
};

export const deleteAllDocuments = async () => {
  const response = await API.delete("/documents");
  return response.data;
};