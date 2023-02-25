import { forwardRef } from 'react';
import './ResultCard.css';

const ResultCard = forwardRef(({ files, predictions }, ref) => {
    return (
        <div ref={ref}>
            {files &&
                predictions &&
                files.length > 0 &&
                predictions.length > 0 &&
                files.map((file, idx) => (
                    <div className='analyzedCard' key={`card-${idx}`}>
                        <div id='resultImages'>
                            <img
                                src={`data:image/jpeg;base64,${file.getFileEncodeBase64String()}`}
                                alt='Uploaded'
                            />
                            {/* <img src="logo192.png" alt="Resulting Groundtruth" /> */}
                        </div>
                        <div>
                            <h1 className={`prediction ${predictions[idx].isTampered === true ? 'tampered' : ''}`}>
                                {predictions[idx].isTampered === true ? 'Tampered' : 'Not Tampered'}
                            </h1>
                            <p className='confidence'>
                                Confidence:&nbsp;
                                {(parseFloat(predictions[idx].confidence) * 100).toFixed(2)}%
                            </p>
                        </div>
                    </div>
                ))}
        </div>
    );
});

export default ResultCard;
