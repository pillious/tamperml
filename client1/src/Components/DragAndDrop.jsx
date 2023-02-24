import React, { useState } from "react";
import FileUploader from "./FileUploader";
import AnalyzeButton from "./AnalyzeButton.jsx";

// Import React FilePond
import { FilePond, registerPlugin } from "react-filepond";

// Import FilePond styles
import "filepond/dist/filepond.min.css";

// Import the Image EXIF Orientation and Image Preview plugins
// Note: These need to be installed separately
// `npm i filepond-plugin-image-preview filepond-plugin-image-exif-orientation --save`
import FilePondPluginImageExifOrientation from "filepond-plugin-image-exif-orientation";
import FilePondPluginImagePreview from "filepond-plugin-image-preview";
import "filepond-plugin-image-preview/dist/filepond-plugin-image-preview.css";

// Register the plugins
registerPlugin(FilePondPluginImageExifOrientation, FilePondPluginImagePreview);

// Our app
const DragAndDrop = (props) => {
  // const [files, setFiles] = useState([]);

  // // Add this function to handle adding files from the FileUploader component
  // const handleAddFile = (file) => {
  //   setFiles([...files, file]);
  // };

  return (
    <div>
      <FilePond
        files={props.files}
        onupdatefiles={props.setFiles}
        allowMultiple={true}
        maxFiles={3}
        server="/api"
        name="files"
        labelIdle='Drag & Drop your files or <span class="filepond--label-action">Browse</span>'
      />
    
        <div className="buttons">
          <FileUploader handleFile={props.handleAddFile} />            
          <AnalyzeButton files={props.files}/>
        </div>
    </div>
  );
};


export default DragAndDrop;
