import './ResultCard.css';

const ResultCard = ({ files, predictions }) => {
    return (
        <>
            {files &&
                predictions &&
                files.length > 0 &&
                predictions.length > 0 &&
                files.map((file, idx) => (
                    <div className='analyzedCard' key={idx}>
                        <h1>
                            {predictions[idx].isTampered === true ? 'Tampered' : 'Not Tampered'}
                        </h1>
                        <p>{(parseFloat(predictions[idx].confidence)*100).toFixed(2)}% Confidence</p>
                        <div id='resultImages'>
                            <img
                                src={`data:image/jpeg;base64,${file.getFileEncodeBase64String()}`}
                                alt="Uploaded"
                            />
                            {/* <img src="logo192.png" alt="Resulting Groundtruth" /> */}
                        </div>
                        <hr></hr>
                    </div>
                ))}
        </>
    );
};

export default ResultCard;
