import { forwardRef } from 'react';
import './ResultCard.css';

const ResultCard = forwardRef(({ predictions }, ref) => {
    return (
        <div ref={ref}>
            {predictions &&
                predictions.length > 0 &&
                predictions.map((pred, idx) => (
                    <div className='analyzedCard' key={`card-${idx}`}>
                        <div id='resultImages'>
                            <img src={`data:image/jpeg;base64,${pred.base64}`} alt='Uploaded' />
                            {/* <img src="logo192.png" alt="Resulting Groundtruth" /> */}
                        </div>
                        <div>
                            <h1
                                className={`prediction ${
                                    pred.isTampered === true ? 'tampered' : ''
                                }`}
                            >
                                {pred.isTampered === true ? 'Tampered' : 'Not Tampered'}
                            </h1>
                            <p className='confidence'>
                                Confidence:&nbsp;
                                {(parseFloat(pred.confidence) * 100).toFixed(2)}%
                            </p>
                        </div>
                    </div>
                ))}
        </div>
    );
});

export default ResultCard;
