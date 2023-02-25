import React, { useState } from 'react';
import './App.css';
import DragAndDrop from './Components/DragAndDrop.jsx';
import ResultCard from './Components/ResultCard/ResultCard.jsx';

function App() {
    const [files, setFiles] = useState([]);
    const [predictions, setPredictions] = useState([]);

    // Add this function to handle adding files from the FileUploader component
    const handleAddFile = (incomingFiles) => {
        // setFiles((prev) => [...prev, ...incomingFiles]);
        console.log(incomingFiles);
        setFiles(incomingFiles);
    };

    const postImage = () => {
        const filesBase64 = files.map((f) => f.getFileEncodeBase64String());
        console.log(filesBase64);

        fetch('http://localhost:5000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ files: filesBase64 }),
        })
            .then((res) => res.json())
            .then((res) => {
                console.log(files);
                console.log(res.data);
                setPredictions(res.data)
            })
            .catch((err) => console.log(err));
    };

    return (
        <div className='App'>
            <header className='App-header'>
                <h1 id='siteName'>TamperML</h1>
                <div className='uploadArea'>
                    <DragAndDrop
                        files={files}
                        handleAddFile={handleAddFile}
                        postImage={postImage}
                    />
                    <ResultCard files={files} predictions={predictions}/>
                </div>
            </header>
        </div>
    );
}

export default App;
