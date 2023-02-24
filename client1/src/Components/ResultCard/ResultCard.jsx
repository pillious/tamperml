import styled from "styled-components";
import "./ResultCard.css";

const ResultCard = () => {
  return (
    <div className="analyzedCard">
      <h1>Likely Tampered</h1>
      <p>95% Confidence Interval</p>
      <div id="resultImages">
        <img src="logo192.png" alt="Uploaded Image" />
        <img src="logo192.png" alt="Resulting Groundtruth" />
      </div>
      <hr></hr>
    </div>
  );
};

export default ResultCard;
