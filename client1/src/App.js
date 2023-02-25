import React, { useRef, useState } from 'react';
import './App.css';
import DragAndDrop from './Components/DragAndDrop.jsx';
import ResultCard from './Components/ResultCard/ResultCard.jsx';

function App() {
    const [files, setFiles] = useState([]);
    const [predictions, setPredictions] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    const ref = useRef(null);

    // Add this function to handle adding files from the FileUploader component
    const handleAddFile = (incomingFiles) => {
        setFiles(incomingFiles);
    };

    const postImage = () => {
        const filesBase64 = files.map((f) => f.getFileEncodeBase64String());

        setIsLoading(true);
        document.body.style.overflow = 'hidden';
        window.scrollTo(0, 0);
        fetch('http://localhost:5000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ files: filesBase64 }),
        })
            .then((res) => res.json())
            .then((res) => {
                setPredictions(res.data);
                setIsLoading(false);
                document.body.style.overflow = 'unset';
                if (ref.current) ref.current.scrollIntoView({ behavior: 'smooth' });
            })
            .catch((err) => {
                console.log(err);
                setIsLoading(false);
                document.body.style.overflow = 'unset';
            });
    };

    return (
        <main>
            <div className='App'>
                <div className={`overlay ${isLoading === true ? '' : 'hide'}`}>
                    <div className='spinner' />
                </div>
                <header className='App-header'>
                    <h1 id='siteName'>TamperML</h1>
                    <div className='uploadArea'>
                        <DragAndDrop
                            files={files}
                            handleAddFile={handleAddFile}
                            postImage={postImage}
                        />
                        <ResultCard files={files} predictions={predictions} ref={ref} />
                    </div>
                </header>
            </div>
        </main>
    );
}

export default App;
