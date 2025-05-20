// frontend/src/components/common/Header.js
// Header component with navigation

import React from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Button, 
  Box 
} from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import TranslateIcon from '@mui/icons-material/Translate';
import SchoolIcon from '@mui/icons-material/School';
import HomeIcon from '@mui/icons-material/Home';

/**
 * Header component with navigation
 * @returns {JSX.Element} Header component
 */
function Header() {
  return (
    <AppBar position="fixed">
      <Toolbar>
        <Typography
          variant="h6"
          component={RouterLink}
          to="/"
          sx={{
            flexGrow: 1,
            color: 'white',
            textDecoration: 'none',
            fontWeight: 'bold'
          }}
        >
          LinguaSign
        </Typography>
        <Box sx={{ display: 'flex' }}>
          <Button 
            color="inherit" 
            component={RouterLink} 
            to="/"
            startIcon={<HomeIcon />}
          >
            Home
          </Button>
          <Button 
            color="inherit" 
            component={RouterLink} 
            to="/translate"
            startIcon={<TranslateIcon />}
          >
            Translate
          </Button>
          <Button 
            color="inherit" 
            component={RouterLink} 
            to="/learn"
            startIcon={<SchoolIcon />}
          >
            Learn
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Header;
