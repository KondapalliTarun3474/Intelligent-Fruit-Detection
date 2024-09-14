import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import MainContent from './components/MainContent';

const App = () => {
  return (
    <div className="app-container">
      <Header />
      <MainContent />
      <Footer />
    </div>
  );
};

export default App;