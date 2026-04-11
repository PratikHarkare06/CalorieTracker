# Calorie Tracker 🍽️📸

## Overview
Calorie Tracker is a modern web application that allows users to upload food images and instantly get nutritional information using the Nutritionix API.

## Features
- 📸 Image upload via drag-and-drop
- 🔍 Automatic food identification
- 📊 Detailed nutritional information
- 🎨 Modern, animated design

## Technologies Used
- React
- Framer Motion
- React Dropzone
- Axios
- Nutritionix API

## Setup Instructions
1. Clone the repository
2. Navigate to the `client` directory
3. Run `npm install`
4. Set up your Nutritionix API credentials
5. Run `npm start`

## Environment Variables
Create a `.env` file in the `client` directory with:
```
REACT_APP_NUTRITIONIX_APP_ID=your_app_id
REACT_APP_NUTRITIONIX_API_KEY=your_api_key
```

## Note
This application requires a Nutritionix API key for food image recognition.
