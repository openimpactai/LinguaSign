// frontend/src/pages/HomePage.js
// Home page component

import React from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  Grid, 
  Card, 
  CardContent, 
  CardActions,
  CardMedia
} from '@mui/material';
import { Link } from 'react-router-dom';
import TranslateIcon from '@mui/icons-material/Translate';
import SchoolIcon from '@mui/icons-material/School';

/**
 * Home page component
 * @returns {JSX.Element} HomePage component
 */
function HomePage() {
  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Hero section */}
      <Box
        sx={{
          bgcolor: 'background.paper',
          pt: 8,
          pb: 6,
          textAlign: 'center'
        }}
      >
        <Typography
          component="h1"
          variant="h2"
          align="center"
          color="text.primary"
          gutterBottom
        >
          LinguaSign
        </Typography>
        <Typography variant="h5" align="center" color="text.secondary" paragraph>
          AI-powered Sign Language Translation and Learning Assistant
        </Typography>
        <Box sx={{ mt: 4 }}>
          <Grid container spacing={2} justifyContent="center">
            <Grid item>
              <Button 
                component={Link} 
                to="/translate" 
                variant="contained" 
                startIcon={<TranslateIcon />}
              >
                Translate Sign Language
              </Button>
            </Grid>
            <Grid item>
              <Button 
                component={Link} 
                to="/learn" 
                variant="outlined" 
                startIcon={<SchoolIcon />}
              >
                Learn Sign Language
              </Button>
            </Grid>
          </Grid>
        </Box>
      </Box>

      {/* Features section */}
      <Grid container spacing={4} sx={{ mt: 4 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardMedia
              component="div"
              sx={{
                pt: '56.25%',
                bgcolor: 'primary.light',
              }}
            />
            <CardContent>
              <Typography gutterBottom variant="h5" component="h2">
                Translation
              </Typography>
              <Typography>
                Translate sign language to text using our AI-powered recognition system.
                Upload videos or use your webcam for real-time translation.
              </Typography>
            </CardContent>
            <CardActions>
              <Button 
                size="small" 
                component={Link} 
                to="/translate"
              >
                Try It
              </Button>
            </CardActions>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardMedia
              component="div"
              sx={{
                pt: '56.25%',
                bgcolor: 'secondary.light',
              }}
            />
            <CardContent>
              <Typography gutterBottom variant="h5" component="h2">
                Learning
              </Typography>
              <Typography>
                Learn sign language with interactive lessons and real-time feedback
                on your gestures. Practice and improve your skills.
              </Typography>
            </CardContent>
            <CardActions>
              <Button 
                size="small" 
                component={Link} 
                to="/learn"
              >
                Start Learning
              </Button>
            </CardActions>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardMedia
              component="div"
              sx={{
                pt: '56.25%',
                bgcolor: 'info.light',
              }}
            />
            <CardContent>
              <Typography gutterBottom variant="h5" component="h2">
                Open Source
              </Typography>
              <Typography>
                LinguaSign is an open-source project. You can contribute to the
                development and help make sign language more accessible.
              </Typography>
            </CardContent>
            <CardActions>
              <Button 
                size="small" 
                href="https://github.com/openimpactai/LinguaSign"
                target="_blank"
                rel="noopener noreferrer"
              >
                View on GitHub
              </Button>
            </CardActions>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default HomePage;
