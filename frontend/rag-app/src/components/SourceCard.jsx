function SourceCard({ source }) {
  return (
    <div>
      <p>{source.content}</p>

      <small>
        Page: {source.page_number}
      </small>

      <small>
        Score: {source.score}
      </small>
    </div>
  );
}

export default SourceCard;