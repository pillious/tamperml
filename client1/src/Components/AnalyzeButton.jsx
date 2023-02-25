import React, { useState, useEffect } from "react";
import styled from "styled-components";

// Style the Button component
const Button = styled.button`
  background-color: ${({ disabled }) => (disabled ? "#D3D3D3" : "#0a1726")};
  color: ${({ disabled }) => (disabled ? "#A9A9A9" : "white")};
  font-size: 2rem;
  font-family: "Imprima", serif;
  padding: 10px 50px;
  font-weight: 300;
  border-color: #0a1726;
  border-radius: 1rem;
  margin: 10px 0px;
  cursor: ${({ disabled }) => (disabled ? "not-allowed" : "pointer")};
`;

const AnalyzeButton = (props) => {
  const [disabled, setDisabled] = useState(true);

  useEffect(() => {
    if (props.files.length > 0) {
      setDisabled(false);
    }
  }, [props.files]);

  const setImagePath = (e) => {
    let reader = new FileReader();
    reader.readAsDataURL(e.target.files[0]);

    reader.onload = () => {
      this.setState({
        queryImage: reader.result,
      });
      setDisabled(false);
    };
  };

  const postImage = () => {
    // set disabled until post request is complete
    setDisabled(true);

    fetch("", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(this.state.queryImage),
    })
      .then((res) => res.json())
      .then((data) => {
        // do stuff with data
        setDisabled(false);
      });
  };

  const handleClick = (event) => {
    if (props.files.length === 0) {
      alert("No valid files uploaded");
    } else {
      alert(props.files.toString());
    }
  };

  return (
    <>
      <Button onClick={handleClick} disabled={disabled}>
        Analyze Images
      </Button>
    </>
  );
};

export default AnalyzeButton;
