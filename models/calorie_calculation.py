import csv
import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalorieCalculator:
    def __init__(self, csv_path=None):
        """
        Initialize the CalorieCalculator with a nutrition database
        
        Args:
            csv_path: Path to the CSV file containing food nutrition data
        """
        if csv_path is None:
            # Default path to user's finalcalories.csv file
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'finalcalories.csv')
        
        try:
            self.food_database = self._load_nutrition_database(csv_path)
            logger.info(f"Loaded nutrition database with {len(self.food_database)} items")
        except Exception as e:
            logger.error(f"Failed to load nutrition database: {str(e)}")
            self.food_database = {}
    
    def _load_nutrition_database(self, csv_path):
        """
        Load the nutrition database from a CSV file
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Dictionary mapping food names to their nutritional information
        """
        food_database = {}
        
        try:
            # Use pandas for more robust CSV handling
            df = pd.read_csv(csv_path)
            
            # Convert DataFrame to dictionary
            for _, row in df.iterrows():
                food_name = row['food_name'].lower()
                
                # Check if the density column exists with the specific format in the user's CSV
                density_col = 'Density(g/cm3)' if 'Density(g/cm3)' in df.columns else 'density'
                
                food_database[food_name] = {
                    "density": float(row[density_col]),
                    "calories_per_g": float(row['calories']),  # Already per gram in the user's CSV
                    "protein_per_g": float(row['protein_g']),
                    "carbs_per_g": float(row['carbs_g']),
                    "fat_per_g": float(row['fat_g'])
                }
                
                # Add fiber and sugars if available
                if 'fiber_g' in df.columns:
                    food_database[food_name]["fiber_per_g"] = float(row['fiber_g'])
                if 'sugars_g' in df.columns:
                    food_database[food_name]["sugars_per_g"] = float(row['sugars_g'])
                if 'sodium_mg' in df.columns:
                    food_database[food_name]["sodium_per_g"] = float(row['sodium_mg'])
            
            return food_database
            
        except Exception as e:
            logger.error(f"Error reading nutrition database: {str(e)}")
            raise
    
    def get_closest_food_match(self, detected_food):
        """
        Find the closest matching food in the database for the detected food
        
        Args:
            detected_food: Food name detected by YOLO
            
        Returns:
            Closest matching food name in the database or None if no match
        """
        detected_food = detected_food.lower()
        
        # Direct match
        if detected_food in self.food_database:
            return detected_food
        
        # Check if detected food is part of any food in database
        for food in self.food_database:
            if detected_food in food or food in detected_food:
                logger.info(f"Matched '{detected_food}' to '{food}' in database")
                return food
        
        # No match found
        logger.warning(f"No match found for '{detected_food}' in nutrition database")
        return None
    
    def volume_to_weight(self, food_name, volume_cm3):
        """
        Convert volume to weight using density
        
        Args:
            food_name: Name of the food
            volume_cm3: Volume in cubic centimeters
            
        Returns:
            Weight in grams
        """
        food_match = self.get_closest_food_match(food_name)
        
        if not food_match or food_match not in self.food_database:
            logger.warning(f"Cannot convert volume to weight: '{food_name}' not found in database")
            # Use a default density of 1.0 g/cm³ if food not found
            return volume_cm3 * 1.0
        
        # Get density from the CSV file for this specific food
        density = self.food_database[food_match]["density"]
        
        # Calculate weight using the formula: weight = volume * density
        weight_grams = volume_cm3 * density
        
        logger.info(f"Weight calculation for {food_name} (matched to {food_match}):") 
        logger.info(f"  Volume: {volume_cm3:.2f} cm³")
        logger.info(f"  Density from CSV: {density:.2f} g/cm³")
        logger.info(f"  Weight = Volume * Density = {volume_cm3:.2f} * {density:.2f} = {weight_grams:.2f}g")
        
        return weight_grams
    
    def calculate_nutrition(self, food_name, volume_cm3):
        """
        Calculate nutritional information based on food volume
        
        Args:
            food_name: Name of the food
            volume_cm3: Volume in cubic centimeters
            
        Returns:
            Dictionary with nutritional information or None if food not found
        """
        food_match = self.get_closest_food_match(food_name)
        
        if not food_match:
            return None
        
        # Convert volume to weight
        weight_grams = self.volume_to_weight(food_match, volume_cm3)
        
        # Get nutritional information
        nutrient = self.food_database[food_match]
        
        # Create base nutrition dictionary
        nutrition = {
            "food_name": food_name,
            "matched_food": food_match,
            "volume_cm3": round(volume_cm3, 1),
            "weight_grams": round(weight_grams, 1),
            "calories": round(nutrient["calories_per_g"] * weight_grams, 1),
            "protein": round(nutrient["protein_per_g"] * weight_grams, 1),
            "carbohydrates": round(nutrient["carbs_per_g"] * weight_grams, 1),
            "fat": round(nutrient["fat_per_g"] * weight_grams, 1)
        }
        
        # Add additional nutritional information if available
        if "fiber_per_g" in nutrient:
            nutrition["fiber"] = round(nutrient["fiber_per_g"] * weight_grams, 1)
        
        if "sugars_per_g" in nutrient:
            nutrition["sugars"] = round(nutrient["sugars_per_g"] * weight_grams, 1)
            
        if "sodium_per_g" in nutrient:
            nutrition["sodium"] = round(nutrient["sodium_per_g"] * weight_grams, 1)
            
        return nutrition
    
    def calculate_nutrition_for_detections(self, food_volumes):
        """
        Calculate nutritional information for multiple food detections
        
        Args:
            food_volumes: List of (food_class, volume) tuples
            
        Returns:
            List of dictionaries with nutritional information
        """
        nutrition_results = []
        total_nutrition = {
            "weight_grams": 0,
            "calories": 0,
            "protein": 0,
            "carbohydrates": 0,
            "fat": 0,
            "fiber": 0,
            "sugars": 0,
            "sodium": 0
        }
        
        for food_class, volume in food_volumes:
            nutrition = self.calculate_nutrition(food_class, volume)
            
            if nutrition:
                nutrition_results.append(nutrition)
                
                # Add to totals
                total_nutrition["weight_grams"] += nutrition["weight_grams"]
                total_nutrition["calories"] += nutrition["calories"]
                total_nutrition["protein"] += nutrition["protein"]
                total_nutrition["carbohydrates"] += nutrition["carbohydrates"]
                total_nutrition["fat"] += nutrition["fat"]
                
                # Add additional nutritional information if available
                if "fiber" in nutrition:
                    total_nutrition["fiber"] += nutrition["fiber"]
                if "sugars" in nutrition:
                    total_nutrition["sugars"] += nutrition["sugars"]
                if "sodium" in nutrition:
                    total_nutrition["sodium"] += nutrition["sodium"]
        
        # Round totals
        for key in total_nutrition:
            total_nutrition[key] = round(total_nutrition[key], 1)
        
        return nutrition_results, total_nutrition

    def calculate_bmi(self, weight_kg, height_cm):
        """
        Calculate BMI using weight in kg and height in cm
        
        Args:
            weight_kg: Weight in kilograms
            height_cm: Height in centimeters
            
        Returns:
            BMI value as a float
        """
        if height_cm <= 0 or weight_kg <= 0:
            logger.error("Height and weight must be positive values")
            return None
            
        # Convert height from cm to meters
        height_m = height_cm / 100
        
        # BMI formula: weight(kg) / height(m)²
        bmi = weight_kg / (height_m * height_m)
        
        logger.info(f"BMI calculation: {weight_kg}kg / ({height_m}m)² = {bmi:.2f}")
        return bmi
    
    def get_bmi_category(self, bmi):
        """
        Get the BMI category and advice based on BMI value
        
        Args:
            bmi: BMI value
            
        Returns:
            Dictionary with category and advice
        """
        bmi_categories = {
            "underweight": {"range": (0, 18.5), "advice": "You may need to increase your calorie intake."},
            "normal": {"range": (18.5, 25), "advice": "Your weight is in the healthy range."},
            "overweight": {"range": (25, 30), "advice": "Consider moderating your calorie intake."},
            "obese": {"range": (30, float('inf')), "advice": "Please consult with a healthcare professional about your diet."}
        }
        
        if bmi is None:
            return {"category": "unknown", "advice": "Invalid BMI calculation"}
            
        for category, data in bmi_categories.items():
            min_val, max_val = data["range"]
            if min_val <= bmi < max_val:
                return {
                    "category": category,
                    "advice": data["advice"]
                }
                
        return {"category": "unknown", "advice": "Could not determine BMI category"}
    
    def calculate_daily_calorie_needs(self, weight_kg, height_cm, age, gender, activity_level="moderate"):
        """
        Calculate estimated daily calorie needs using the Harris-Benedict equation
        
        Args:
            weight_kg: Weight in kilograms
            height_cm: Height in centimeters
            age: Age in years
            gender: 'male' or 'female'
            activity_level: Activity level (sedentary, light, moderate, active, very_active)
            
        Returns:
            Estimated daily calorie needs
        """
        # Activity level multipliers
        activity_multipliers = {
            "sedentary": 1.2,  # Little or no exercise
            "light": 1.375,    # Light exercise 1-3 days/week
            "moderate": 1.55,  # Moderate exercise 3-5 days/week
            "active": 1.725,   # Hard exercise 6-7 days/week
            "very_active": 1.9 # Very hard exercise & physical job or training twice a day
        }
        
        if gender.lower() == 'male':
            # Men: BMR = 88.362 + (13.397 × weight in kg) + (4.799 × height in cm) - (5.677 × age in years)
            bmr = 88.362 + (13.397 * weight_kg) + (4.799 * height_cm) - (5.677 * age)
        else:
            # Women: BMR = 447.593 + (9.247 × weight in kg) + (3.098 × height in cm) - (4.330 × age in years)
            bmr = 447.593 + (9.247 * weight_kg) + (3.098 * height_cm) - (4.330 * age)
        
        # Get activity multiplier (default to moderate if invalid)
        multiplier = activity_multipliers.get(activity_level.lower(), activity_multipliers["moderate"])
        
        # Calculate total daily energy expenditure
        daily_calories = bmr * multiplier
        
        logger.info(f"Daily calorie needs calculation for {gender}, {age} years, {activity_level} activity:")
        logger.info(f"  BMR: {bmr:.2f} calories")
        logger.info(f"  Activity multiplier: {multiplier}")
        logger.info(f"  Daily calorie needs: {daily_calories:.2f} calories")
        
        return daily_calories
    
    def calculate_meal_calorie_distribution(self, daily_calories):
        """
        Calculate recommended calorie distribution for meals
        
        Args:
            daily_calories: Total daily calorie needs
            
        Returns:
            Dictionary with meal calorie recommendations
        """
        return {
            "breakfast": round(daily_calories * 0.25),  # 25% of daily calories
            "lunch": round(daily_calories * 0.35),      # 35% of daily calories
            "dinner": round(daily_calories * 0.30),     # 30% of daily calories
            "snacks": round(daily_calories * 0.10)      # 10% of daily calories
        }
    
    def analyze_meal_calories(self, meal_calories, daily_calorie_needs):
        """
        Analyze if the meal calories are within recommended range
        
        Args:
            meal_calories: Calories in the current meal
            daily_calorie_needs: Total daily calorie needs
            
        Returns:
            Dictionary with analysis results
        """
        # Estimate single meal calorie intake (assuming 3 main meals + snacks)
        avg_meal_calories = daily_calorie_needs / 3
        
        # Calculate reasonable range (±20% of average meal calories)
        lower_bound = avg_meal_calories * 0.8
        upper_bound = avg_meal_calories * 1.2
        
        if meal_calories < lower_bound:
            status = "below"
            message = f"This meal ({meal_calories} kcal) is below your estimated single meal intake of ~{round(avg_meal_calories)} kcal."
        elif meal_calories > upper_bound:
            status = "above"
            message = f"This meal ({meal_calories} kcal) is above your estimated single meal intake of ~{round(avg_meal_calories)} kcal."
        else:
            status = "within"
            message = f"This meal ({meal_calories} kcal) is within your estimated single meal intake of ~{round(avg_meal_calories)} kcal."
            
        return {
            "status": status,
            "message": message,
            "meal_calories": meal_calories,
            "recommended_meal_calories": round(avg_meal_calories),
            "daily_calorie_needs": round(daily_calorie_needs)
        }

# For testing
if __name__ == "__main__":
    calculator = CalorieCalculator()
    print(calculator.calculate_nutrition("apple", 100))
    
    # Test BMI calculation
    bmi = calculator.calculate_bmi(70, 175)
    print(f"BMI: {bmi:.2f}")
    print(calculator.get_bmi_category(bmi))
    
    # Test calorie needs calculation
    daily_calories = calculator.calculate_daily_calorie_needs(70, 175, 30, 'male', 'moderate')
    print(f"Daily calorie needs: {daily_calories:.2f}")
    print(calculator.calculate_meal_calorie_distribution(daily_calories))
