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
  const setImagePath = (e) => {
    let reader = new FileReader();
    reader.readAsDataURL(e.target.files[0]);

    reader.onload = () => {
      this.setState({
        queryImage: reader.result,
      });
    };
  };

  const postImage = () => {
    fetch("", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(this.state.queryImage),
    })
      .then((res) => res.json())
      .then((data) => {
        // do stuff with data
      });
  };

  const handleClick = (event) => {
    if ((props.files).length == 0) {
      alert("No valid files uploaded")
    } else {
      alert((props.files).toString);
    }
    
  };
  return (
    <>
      <Button onClick={handleClick}>Analyze Images</Button>
    </>
  );
};

export default AnalyzeButton;
