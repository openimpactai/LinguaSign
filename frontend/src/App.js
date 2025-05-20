// frontend/src/App.js
// Main application component

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Container, CssBaseline, ThemeProvider, createTheme } from '@mui/material';

// Import pages
import HomePage from './pages/HomePage';
import TranslatePage from './pages/TranslatePage';
import LearnPage from './pages/LearnPage';

// Import components
import Header from './components/common/Header';
import Footer from './components/common/Footer';

// Create theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#3f51b5',
    },
    secondary: {
      main: '#f50057',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Header />
        <Container component="main" sx={{ mt: 8, mb: 4 }}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/translate" element={<TranslatePage />} />
            <Route path="/learn" element={<LearnPage />} />
          </Routes>
        </Container>
        <Footer />
      </Router>
    </ThemeProvider>
  );
}

export default App;
