import "./App.css";
import FileUploader from "./Components/FileUploader.jsx";
import DragAndDrop from "./Components/DragAndDrop.jsx";
import AnalyzeButton from "./Components/AnalyzeButton.jsx";
import ResultCard from "./Components/ResultCard/ResultCard.jsx";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1 id="siteName">TamperML</h1>
        <div className="uploadArea">
          <DragAndDrop />
          <div className="buttons">
            <FileUploader />
            <AnalyzeButton />
          </div>
          <ResultCard />
        </div>
      </header>
    </div>
  );
}

export default App;
