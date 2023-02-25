import React, { useRef } from "react";
import styled from "styled-components";

const Button = styled.button`
  background-color: #0a1726;
  color: white;
  font-size: 2rem;
  font-family: "Imprima", serif;
  padding: 10px 50px;
  font-weight: 300;
  border-color: #0a1726;
  border-radius: 1rem;
  margin: 10px 0px;
  cursor: pointer;
`;

const FileUploader = (props) => {
  const hiddenFileInput = useRef(null);

  const handleClick = () => {
    hiddenFileInput.current.click();
  };

  const handleChange = (event) => {
    const filesUploaded = event.target.files;
    const filesArray = [];
    for (const f of filesUploaded) {
      filesArray.push(f);
    }
    props.handleFile(filesArray); // Call the handleFile function passed as a prop
  };

  return (
    <>
      <Button onClick={handleClick}>Upload File(s)</Button>
      <input
        type="file"
        multiple
        ref={hiddenFileInput}
        onChange={handleChange}
        style={{ display: "none" }}
      />
    </>
  );
};

export default FileUploader;
