import SourceCard from "./SourceCard";

function Message({ msg }) {
  return (
    <div>
      <p>
        <b>{msg.role}</b>: {msg.text}
      </p>

      {msg.sources &&
        msg.sources.map((source, index) => (
          <SourceCard
            key={index}
            source={source}
          />
        ))}
    </div>
  );
}

export default Message;