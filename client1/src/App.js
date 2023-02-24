import "./App.css";
import DragAndDrop from "./Components/DragAndDrop.jsx";
import ResultCard from "./Components/ResultCard/ResultCard.jsx";
import React, { useState } from "react";

function App() {
  const [files, setFiles] = useState([]);

  // Add this function to handle adding files from the FileUploader component
  const handleAddFile = (file) => {
    setFiles([...files, file]);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1 id="siteName">TamperML</h1>
        <div className="uploadArea">
          <DragAndDrop files={files} setFiles={setFiles} handleAddFile={handleAddFile}/>
          <ResultCard />
        </div>
      </header>
    </div>
  );
}

export default App;
