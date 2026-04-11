import logging
from .calorie_calculation import CalorieCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthAdvisor:
    def __init__(self, calorie_calculator=None):
        """
        Initialize the HealthAdvisor with a CalorieCalculator instance
        
        Args:
            calorie_calculator: CalorieCalculator instance
        """
        if calorie_calculator is None:
            self.calculator = CalorieCalculator()
        else:
            self.calculator = calorie_calculator
    
    def get_user_health_data(self):
        """
        Get health-related data from user input
        
        Returns:
            Dictionary with user health data
        """
        try:
            weight_kg = float(input("Enter your weight in kg: "))
            height_cm = float(input("Enter your height in cm: "))
            age = int(input("Enter your age in years: "))
            
            gender = ""
            while gender not in ["male", "female"]:
                gender = input("Enter your gender (male/female): ").lower()
                if gender not in ["male", "female"]:
                    print("Please enter either 'male' or 'female'")
            
            activity_levels = {
                "1": "sedentary",
                "2": "light",
                "3": "moderate",
                "4": "active",
                "5": "very_active"
            }
            
            print("\nActivity Levels:")
            print("1 - Sedentary (little or no exercise)")
            print("2 - Light (exercise 1-3 days/week)")
            print("3 - Moderate (exercise 3-5 days/week)")
            print("4 - Active (exercise 6-7 days/week)")
            print("5 - Very Active (hard exercise & physical job)")
            
            activity_choice = ""
            while activity_choice not in activity_levels:
                activity_choice = input("Select your activity level (1-5): ")
                if activity_choice not in activity_levels:
                    print("Please enter a number between 1 and 5")
            
            activity_level = activity_levels[activity_choice]
            
            return {
                "weight_kg": weight_kg,
                "height_cm": height_cm,
                "age": age,
                "gender": gender,
                "activity_level": activity_level
            }
            
        except ValueError as e:
            logger.error(f"Error getting user input: {str(e)}")
            print("Please enter valid numerical values for weight, height, and age.")
            return self.get_user_health_data()  # Retry
    
    def provide_health_advice(self, total_calories=None):
        """
        Provide health advice based on user data and detected food calories
        
        Args:
            total_calories: Total calories from detected food (optional)
            
        Returns:
            Dictionary with health advice
        """
        # Get user health data
        user_data = self.get_user_health_data()
        
        # Calculate BMI
        bmi = self.calculator.calculate_bmi(user_data["weight_kg"], user_data["height_cm"])
        bmi_category = self.calculator.get_bmi_category(bmi)
        
        # Calculate daily calorie needs
        daily_calorie_needs = self.calculator.calculate_daily_calorie_needs(
            user_data["weight_kg"],
            user_data["height_cm"],
            user_data["age"],
            user_data["gender"],
            user_data["activity_level"]
        )
        
        # Calculate meal distribution
        meal_distribution = self.calculator.calculate_meal_calorie_distribution(daily_calorie_needs)
        
        # Prepare advice
        advice = {
            "user_data": user_data,
            "bmi": round(bmi, 2),
            "bmi_category": bmi_category["category"],
            "bmi_advice": bmi_category["advice"],
            "daily_calorie_needs": round(daily_calorie_needs),
            "meal_distribution": meal_distribution
        }
        
        # Add meal analysis if total calories provided
        if total_calories is not None:
            meal_analysis = self.calculator.analyze_meal_calories(total_calories, daily_calorie_needs)
            advice["meal_analysis"] = meal_analysis
        
        return advice
    
    def display_health_advice(self, total_calories=None):
        """
        Display health advice to the user
        
        Args:
            total_calories: Total calories from detected food (optional)
        """
        advice = self.provide_health_advice(total_calories)
        
        print("\n" + "="*50)
        print("YOUR HEALTH PROFILE")
        print("="*50)
        
        print(f"\nBMI: {advice['bmi']}")
        print(f"BMI Category: {advice['bmi_category'].title()}")
        print(f"BMI Advice: {advice['bmi_advice']}")
        
        print("\n" + "-"*50)
        print("DAILY CALORIE RECOMMENDATIONS")
        print("-"*50)
        
        print(f"\nEstimated Daily Calorie Needs: {advice['daily_calorie_needs']} kcal")
        print("\nRecommended Meal Distribution:")
        print(f"  Breakfast: {advice['meal_distribution']['breakfast']} kcal")
        print(f"  Lunch: {advice['meal_distribution']['lunch']} kcal")
        print(f"  Dinner: {advice['meal_distribution']['dinner']} kcal")
        print(f"  Snacks: {advice['meal_distribution']['snacks']} kcal")
        
        if total_calories is not None:
            print("\n" + "-"*50)
            print("CURRENT MEAL ANALYSIS")
            print("-"*50)
            
            print(f"\nDetected Meal Calories: {total_calories} kcal")
            print(f"Status: {advice['meal_analysis']['message']}")
            
            # Additional advice based on BMI and meal calories
            if advice['bmi_category'] == 'overweight' or advice['bmi_category'] == 'obese':
                if advice['meal_analysis']['status'] == 'above':
                    print("\nSuggestion: Consider reducing portion sizes or choosing lower-calorie alternatives.")
                elif advice['meal_analysis']['status'] == 'within':
                    print("\nSuggestion: This meal is appropriate for your calorie needs, but consider increasing physical activity.")
                else:
                    print("\nSuggestion: This meal is within your calorie goals. Focus on nutrient-dense foods and regular exercise.")
            elif advice['bmi_category'] == 'underweight':
                if advice['meal_analysis']['status'] == 'below':
                    print("\nSuggestion: Consider increasing portion sizes or adding calorie-dense, nutritious foods.")
                else:
                    print("\nSuggestion: This meal is good for helping you reach a healthier weight.")
            else:  # normal weight
                print("\nSuggestion: Maintain your balanced diet and regular physical activity.")
        
        print("\n" + "="*50)


# For testing
if __name__ == "__main__":
    advisor = HealthAdvisor()
    
    # Test with a sample meal of 600 calories
    advisor.display_health_advice(600)
