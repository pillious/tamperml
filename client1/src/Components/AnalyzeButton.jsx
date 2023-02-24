import React from "react";
import styled from "styled-components";

// Style the Button component
const Button = styled.button`
  background-color: #0a1726;
  color: white;
  font-size: 2rem;
  padding: 10px 60px;
  border-color: #0a1726;
  border-radius: 1rem;
  margin: 10px 0px;
  cursor: pointer;
`;

const AnalyzeButton = (props) => {
  const handleClick = (event) => {
    alert("Handle Click");
  };
  return (
    <>
      <Button onClick={handleClick}>Analyze Images</Button>
    </>
  );
};

export default AnalyzeButton;
