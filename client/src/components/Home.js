import React, { useState } from 'react'
import './Home.css'

const Home = () => {

    const[hours_studied,setHoursStudies] = useState('');
    const[previous_score,setPrevScore] = useState('');
    const[extracurricular_activities,setExtra]=useState('');
    const[sleep_hours,setSleepHours] = useState('');
    const[sample_papers_solved,setSamplePaper] = useState('');

    const [predictedScore,setPredictedScore] = useState('');

    const handlePrediction = async () => {
        const response = await fetch('http://127.0.0.1:5000/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            hours_studied,
            previous_score,
            extracurricular_activities,
            sleep_hours,
            sample_papers_solved
          }),
        });
    
        if (response.ok) {
          const data = await response.json();
          setPredictedScore(data.predicted_score);
        }
      };

  return (
        <div className="full_page">
           <div className='centre-top'> <h1 id="main_heading" align="center">Student Score Predictor</h1></div>
            <div className="centre_div">
                <label htmlFor="">Hours Studied</label>
                <input type="number" onChange={(e)=>setHoursStudies(e.target.value)}/>
                <br />
                <label htmlFor="">Previous Score</label>
                <input type="number" onChange={(e)=>setPrevScore(e.target.value)}/>
                <br />
                <label htmlFor="">Giving time to Extracurricular?</label>
                <select name="" id="" onChange = {(e)=>setExtra(e.target.value)}>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
                <br />
                <label htmlFor="">Hours slept</label>
                <input type="number" onChange={(e)=>setSleepHours(e.target.value)}/>
                <br />
                <label htmlFor="">Sample Papers Solved</label>
                <input type="number" onChange={(e)=>setSamplePaper(e.target.value)}/>
                <br />
                <button id="submit_btn" onClick={handlePrediction}>Submit</button>
                <br />
                <h2>YOUR EXPECTED SCORE :{predictedScore} </h2>
            </div>
        </div>
  )
}

export default Home
