import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './App.css';

// Placeholder components for the new pages
const ImageUploadPage = ({ setCurrentPage, setImageData }) => {
  const [topImage, setTopImage] = useState(null);
  const [sideImage, setSideImage] = useState(null);
  const [topImagePreview, setTopImagePreview] = useState(null);
  const [sideImagePreview, setSideImagePreview] = useState(null);

  const handleImageChange = (e, setImage, setPreview) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
    }
  };

  const handleNext = () => {
    if (topImage && sideImage) {
      setImageData({ top: topImage, side: sideImage });
      setCurrentPage('foodRecognition');
    } else {
      alert('Please upload both images.');
    }
  };

  return (
    <motion.div 
      className="page-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      <h2>📸 Upload Meal Images</h2>
      
      <label htmlFor="top-image-input" className="custom-file-upload">
        Top-View Image
      </label>
      <input 
        type="file" 
        id="top-image-input" 
        accept="image/*" 
        onChange={(e) => handleImageChange(e, setTopImage, setTopImagePreview)} 
      />
      {topImagePreview && (
        <div className="thumbnail-container">
          <img src={topImagePreview} alt="Top view preview" className="thumbnail"/>
        </div>
      )}
      
      <label htmlFor="side-image-input" className="custom-file-upload">
        Side-View Image
      </label>
      <input 
        type="file" 
        id="side-image-input" 
        accept="image/*" 
        onChange={(e) => handleImageChange(e, setSideImage, setSideImagePreview)} 
      />
      {sideImagePreview && (
        <div className="thumbnail-container">
          <img src={sideImagePreview} alt="Side view preview" className="thumbnail"/>
        </div>
      )}
      
      <motion.button 
        onClick={handleNext} 
        whileHover={{ scale: 1.05 }} 
        whileTap={{ scale: 0.95 }}
        className="button-primary" // Assuming you have a primary button style
      >
        Next
      </motion.button>
    </motion.div>
  );
};

const FoodRecognitionPage = ({ setCurrentPage, imageData, setMealData }) => {
  // Placeholder for food recognition and calorie estimation
  const [recognizedFood, setRecognizedFood] = useState([]);

  React.useEffect(() => {
    if (imageData) {
      console.log('Processing images:', imageData);
      setTimeout(() => {
        const mockFoodData = [
          { id: 1, name: 'Apple', volume: '150ml', calories: '95kcal', region: 'top-left' },
          { id: 2, name: 'Banana', volume: '100ml', calories: '105kcal', region: 'bottom-right' },
          { id: 3, name: 'Orange Juice', volume: '250ml', calories: '112kcal', region: 'center' },
        ];
        setRecognizedFood(mockFoodData);
        setMealData(mockFoodData);
      }, 1500);
    }
  }, [imageData, setMealData]);

  const listVariants = {
    visible: { 
      opacity: 1,
      transition: { staggerChildren: 0.2 }
    },
    hidden: { opacity: 0 }
  };

  const itemVariants = {
    visible: { opacity: 1, y: 0 },
    hidden: { opacity: 0, y: 20 }
  };

  return (
    <motion.div className="page-container">
      <h2>🍱 Food Recognition & Calorie Estimation</h2>
      {imageData && <p>Images received. Processing...</p>}
      {recognizedFood.length > 0 ? (
        <motion.div 
          className="food-results"
          variants={listVariants}
          initial="hidden"
          animate="visible"
        >
          {recognizedFood.map(food => (
            <motion.div key={food.id} className="food-item-card" variants={itemVariants}>
              <p><strong>{food.name}</strong></p>
              <p>Volume: {food.volume}</p>
              <p>Calories: {food.calories}</p>
              {/* Add region marker display here if needed */}
            </motion.div>
          ))}
          <motion.button 
            onClick={() => setCurrentPage('userInfo')}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Next
          </motion.button>
        </motion.div>
      ) : (
        <p>Recognizing food items... Please wait.</p>
      )}
    </motion.div>
  );
};

const UserInfoPage = ({ mealData }) => {
  const [userInfo, setUserInfo] = useState({ name: '', age: '', gender: 'male', height: '', weight: '' });
  const [bmi, setBmi] = useState(null);
  const [bmiCategory, setBmiCategory] = useState('');
  const [advice, setAdvice] = useState('');
  const [showResults, setShowResults] = useState(false);

  const handleChange = (e) => {
    setUserInfo({ ...userInfo, [e.target.name]: e.target.value });
  };

  const calculateBmiAndAdvice = () => {
    const heightM = parseFloat(userInfo.height) / 100;
    const weightKg = parseFloat(userInfo.weight);
    const age = parseInt(userInfo.age);

    if (heightM > 0 && weightKg > 0 && age > 0) {
      const bmiValue = weightKg / (heightM * heightM);
      setBmi(bmiValue.toFixed(2));

      let category = '';
      if (bmiValue < 18.5) category = 'Underweight';
      else if (bmiValue < 24.9) category = 'Normal weight';
      else if (bmiValue < 29.9) category = 'Overweight';
      else category = 'Obese';
      setBmiCategory(category);
      
      let bmr;
      if (userInfo.gender === 'male') {
        bmr = (10 * weightKg) + (6.25 * parseFloat(userInfo.height)) - (5 * age) + 5;
      } else {
        bmr = (10 * weightKg) + (6.25 * parseFloat(userInfo.height)) - (5 * age) - 161;
      }
      const recommendedDailyIntake = bmr * 1.2; 

      const totalMealCalories = mealData ? mealData.reduce((sum, item) => sum + parseInt(item.calories), 0) : 0;
      const recommendedMealIntake = recommendedDailyIntake / 3;

      if (totalMealCalories <= recommendedMealIntake) {
        setAdvice(`✅ This meal (${totalMealCalories} kcal) is within your estimated single meal intake of ~${recommendedMealIntake.toFixed(0)} kcal. Your estimated daily needs: ~${recommendedDailyIntake.toFixed(0)} kcal.`);
      } else {
        setAdvice(`⚠️ Warning: This meal (${totalMealCalories} kcal) may exceed your estimated single meal intake of ~${recommendedMealIntake.toFixed(0)} kcal. Your estimated daily needs: ~${recommendedDailyIntake.toFixed(0)} kcal.`);
      }
      
      setShowResults(true);
    } else {
      alert('Please enter valid age, height, and weight.');
    }
  };
  
  return (
    <div className="user-info-page">
      <h2>🧍‍♂️ User Info & Calorie Evaluation</h2>
      
      <div className="user-form">
        <div className="input-group">
          <input 
            type="text" 
            name="name" 
            placeholder="Name" 
            onChange={handleChange} 
            value={userInfo.name} 
          />
        </div>
        
        <div className="input-group">
          <input 
            type="number" 
            name="age" 
            placeholder="Age (years)" 
            onChange={handleChange} 
            value={userInfo.age} 
            min="1" 
          />
        </div>
        
        <div className="input-group">
          <select 
            name="gender" 
            onChange={handleChange} 
            value={userInfo.gender}
          >
            <option value="male">Male</option>
            <option value="female">Female</option>
          </select>
        </div>
        
        <div className="input-group">
          <input 
            type="number" 
            name="height" 
            placeholder="Height (cm)" 
            onChange={handleChange} 
            value={userInfo.height} 
            min="1" 
          />
        </div>
        
        <div className="input-group">
          <input 
            type="number" 
            name="weight" 
            placeholder="Weight (kg)" 
            onChange={handleChange} 
            value={userInfo.weight} 
            min="1" 
          />
        </div>
        
        <div className="button-container">
          <button 
            className="calculate-button"
            onClick={calculateBmiAndAdvice}
          >
            Calculate BMI & Get Advice
          </button>
        </div>
      </div>
      
      {showResults && (
        <div className="results-box">
          <h3>Your Results:</h3>
          <p><strong>BMI:</strong> {bmi} ({bmiCategory})</p>
          <p>{advice}</p>
        </div>
      )}
    </div>
  );
};

function App() {
  const [currentPage, setCurrentPage] = useState('imageUpload'); // imageUpload, foodRecognition, userInfo
  const [imageData, setImageData] = useState(null); // To store uploaded image data
  const [mealData, setMealData] = useState(null); // To store recognized food data
  const [theme, setTheme] = useState(() => {
    const savedTheme = localStorage.getItem('theme');
    return savedTheme ? savedTheme : 'light'; // Default to light theme
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  }, []);

  // Animation variants
  const pageVariants = {
    initial: { opacity: 0, x: "-100%" },
    in: { opacity: 1, x: 0 },
    out: { opacity: 0, x: "100%" }
  };

  const pageTransition = {
    type: "tween",
    ease: "anticipate",
    duration: 0.5
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'imageUpload':
        return <ImageUploadPage setCurrentPage={setCurrentPage} setImageData={setImageData} />;
      case 'foodRecognition':
        return <FoodRecognitionPage setCurrentPage={setCurrentPage} imageData={imageData} setMealData={setMealData} />;
      case 'userInfo':
        return <UserInfoPage mealData={mealData} />;
      default:
        return <ImageUploadPage setCurrentPage={setCurrentPage} setImageData={setImageData} />;
    }
  };

  return (
    <div className="App">
      <div className="container">
        <button onClick={toggleTheme} className="theme-toggle-button">
          {theme === 'light' ? '🌙 Dark Mode' : '☀️ Light Mode'}
        </button>
        <h1>Calorie Tracker</h1>
        <AnimatePresence mode='wait'>
          {renderPage()}
        </AnimatePresence>
      </div>
    </div>
  );
}

export default App;
