import React, { useEffect, useRef } from 'react';
import VideoPlayer from './VideoPlayer';

const MainContent = () => {
  const videoRef = useRef(null);

  return (
    <main className="main-content">
      <div className="box">
        <div className="content-box">
          <div className="camera-section">
            <center>
              <div className="title camera-title">Camera</div>
              <div className="camera-preview">
                <VideoPlayer/>
              </div>
            </center>
          </div>
          <div className="fruits-section">
            <center>
              <div className="title fruits-title">Detected Fruits</div>
              <ul className="fruits-list">
                <li>Apple</li>
                <li>Banana</li>
                <li>Orange</li>
              </ul>
            </center>
          </div>
        </div>
      </div>
    </main>
  );
};

export default MainContent;
