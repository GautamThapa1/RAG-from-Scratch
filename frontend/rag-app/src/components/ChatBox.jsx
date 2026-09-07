import { useState } from "react";
import { askQuestion } from "../services/api";
function ChatBox({ messages, setMessages }) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    const question = input.trim();
    if (!question) return; // prevents empty messages

    const userMessage = {
      role: "user",
      text: question,
    };
    
    // collect then and append new messages in an array         
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    setLoading(true);

    // make the call
    try {
      const data = await askQuestion(question);

      const botMessage = {
        role: "assistant",
        text: data.answer,
        sources: data.sources,
      };
      // append to the array
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.log(error);
    }

    setLoading(false);
  };

  return (
    <div>
      <div>
        {messages.map((msg, index) => (
          <div key={index}>
            <b>{msg.role}</b>: {msg.text}

            {msg.sources &&
              msg.sources.map((source, i) => (
                <div key={i}>
                  Page: {source.page_number}
                </div>
              ))}
          </div>
        ))}
      </div>

      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) =>{
          if (e.key === "Enter") handleSend();
        }}
        placeholder="Ask question..."
        disabled={loading} // so user can't type while fetching the api call
      />

      <button onClick={handleSend}>
        Send
      </button>

      {loading && <p>Thinking...</p>}
    </div>
  );
}

export default ChatBox;