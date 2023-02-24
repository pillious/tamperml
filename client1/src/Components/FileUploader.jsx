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
const FileUploader = (props) => {
  // Create a reference to the hidden file input element
  const hiddenFileInput = React.useRef(null);

  // Programatically click the hidden file input element
  // when the Button component is clicked
  const handleClick = (event) => {
    hiddenFileInput.current.click();
  };
  // Call a function (passed as a prop from the parent component)
  // to handle the user-selected file
  const handleChange = (event) => {
    const fileUploaded = event.target.files[0];
    props.handleFile(fileUploaded);
  };
  return (
    <>
      <Button onClick={handleClick}>Upload File(s)</Button>
      <input
        type="file"
        ref={hiddenFileInput}
        onChange={handleChange}
        style={{ display: "none" }}
      />
    </>
  );
};

export default FileUploader;
