import streamlit as st
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import io
import pandas as pd

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from models.image_detection import ViewDetector
from models.volume_estimation import estimate_food_volume, get_depth, load_midas_model
from models.calorie_calculation import CalorieCalculator
from models.health_advice import HealthAdvisor

st.set_page_config(page_title="Food Detection", layout="wide")

def process_image(uploaded_file, detector):
    if uploaded_file is None:
        return
    
    try:
        # Convert uploaded file to opencv format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Detect food items
        detected_image, detections = detector.detect_food(image)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Detection", "Volume Estimation", "Calorie Calculation", "Health Advice"])
        
        with tab1:
            # Display detection results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), 
                        caption='Detected Food Items')
                
            with col2:
                st.subheader("Detection Results")
                if detections:
                    detection_count = 0
                    segmentation_count = 0
                    
                    for i, det in enumerate(detections, 1):
                        if det.get('type') == 'segmentation':
                            segmentation_count += 1
                            with st.expander(f"Segmentation {segmentation_count}: {det['class']}"):
                                st.write(f"Confidence: {det['confidence']:.2%}")
                                st.write(f"Type: Segmentation Mask")
                                if 'mask_shape' in det:
                                    st.write(f"Mask shape: {det['mask_shape']}")
                        else:  # Default to detection
                            detection_count += 1
                            with st.expander(f"Detection {detection_count}: {det['class']}"):
                                st.write(f"Confidence: {det['confidence']:.2%}")
                                if 'box' in det:
                                    st.write(f"Location: {det['box']}")
                    
                    # Display summary
                    st.info(f"Found {detection_count} object detections and {segmentation_count} segmentation masks")
                else:
                    st.warning("No items detected")
        
        with tab2:
            # Add volume estimation
            st.subheader("Food Volume Estimation")
            
            # Add calibration parameters
            col1, col2 = st.columns(2)
            with col1:
                pixel_to_cm_ratio = st.slider("Pixel to cm ratio", 0.01, 0.5, 0.1, 0.01, 
                                            help="Conversion ratio from pixels to centimeters. Adjust based on camera distance.")
            with col2:
                depth_scale_factor = st.slider("Depth scale factor", 1.0, 20.0, 10.0, 0.5,
                                             help="Factor to scale depth values to centimeters. Higher values = larger volumes.")
            
            # Show depth map visualization toggle
            show_depth_details = st.checkbox("Show depth map details", value=True, 
                                          help="Display the raw depth map and additional visualizations")
            
            # Automatically run volume estimation
            with st.spinner("Estimating food volume... This may take a moment."):
                try:
                    # Get RGB image for processing
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Load depth model first to show progress
                    depth_model, device = load_midas_model()
                    st.info("Depth estimation model loaded successfully")
                    
                    # Generate depth map
                    depth_map = get_depth(rgb_image, depth_model, device)
                    
                    # Display raw depth map if requested
                    if show_depth_details:
                        st.subheader("Depth Map Visualization")
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Original image
                        ax1.imshow(rgb_image)
                        ax1.set_title("Original Image")
                        ax1.axis('off')
                        
                        # Depth map with colormap
                        depth_vis = ax2.imshow(depth_map, cmap='plasma')
                        ax2.set_title("Depth Map")
                        ax2.axis('off')
                        fig.colorbar(depth_vis, ax=ax2, label='Depth')
                        
                        # Save figure to buffer and display
                        depth_buf = io.BytesIO()
                        plt.tight_layout()
                        plt.savefig(depth_buf, format='png')
                        depth_buf.seek(0)
                        st.pyplot(fig)
                        
                        # Display depth statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Min Depth", f"{np.min(depth_map):.3f}")
                        with col2:
                            st.metric("Max Depth", f"{np.max(depth_map):.3f}")
                        with col3:
                            st.metric("Mean Depth", f"{np.mean(depth_map):.3f}")
                    
                    # Estimate food volumes using the already generated depth map
                    volumes, fig_buf = estimate_food_volume(
                        image=image,
                        yolo_model=detector.model,
                        pixel_to_cm_ratio=pixel_to_cm_ratio,
                        depth_scale_factor=depth_scale_factor,
                        yolo_results=detector.last_results  # Pass the stored YOLO results
                    )
                    
                    if volumes and fig_buf:
                        # Display results
                        st.subheader("Volume Estimation Results")
                        
                        # Display visualization
                        img = Image.open(fig_buf)
                        st.image(img, caption="Volume Estimation Visualization", use_column_width=True)
                        
                        # Display volume table
                        volume_data = []
                        total_volume = 0
                        for i, (food_class, volume) in enumerate(volumes):
                            volume_data.append({
                                "Item": i+1,
                                "Food Class": food_class,
                                "Volume (cm³)": f"{volume:.2f}"
                            })
                            total_volume += volume
                        
                        st.table(volume_data)
                        
                        # Show total volume
                        st.success(f"Total estimated food volume: {total_volume:.2f} cm³")
                        
                        # Store volumes for calorie calculation
                        st.session_state.food_volumes = volumes
                        
                        # Add explanation of volume calculation
                        with st.expander("How volume is calculated"):
                            st.write("""
                            Volume is calculated using the depth map and segmentation masks:
                            1. For each pixel in the food item, we get its depth value
                            2. Each pixel represents a small volume: pixel area (cm²) × depth (cm)
                            3. The total volume is the sum of all pixel volumes
                            
                            Adjust the pixel-to-cm ratio based on your camera setup, and the depth scale factor to calibrate the volume estimates.
                            """)
                    else:
                        st.warning("Could not estimate volumes. Please try adjusting the calibration parameters.")
                except Exception as e:
                    st.error(f"Error during volume estimation: {str(e)}")
                    st.info("Volume estimation works with both segmentation masks and bounding boxes.")
                    
            # Add recalculation button for convenience
            if st.button("Recalculate with Current Parameters"):
                st.rerun()
                
        # Calorie Calculation Tab
        with tab3:
            st.subheader("Calorie Calculation")
            
            if 'food_volumes' not in st.session_state or not st.session_state.food_volumes:
                st.warning("Please complete volume estimation first to calculate calories")
            else:
                try:
                    # Initialize calorie calculator
                    calorie_calculator = CalorieCalculator()
                    
                    # Calculate nutrition for all detected food items
                    nutrition_results, total_nutrition = calorie_calculator.calculate_nutrition_for_detections(
                        st.session_state.food_volumes
                    )
                    
                    # Display nutrition results
                    if nutrition_results:
                        # Display nutrition table
                        st.subheader("Nutritional Information")
                        
                        # Create a dataframe for better display
                        nutrition_data = []
                        
                        # First, show the weight calculation details
                        st.subheader("Weight Calculation from Volume")
                        weight_calc_data = []
                        
                        for item in nutrition_results:
                            # Get the density used for this food item
                            density = calorie_calculator.food_database.get(item["matched_food"], {}).get("density", 1.0)
                            
                            weight_calc_data.append({
                                "Food Item": item["food_name"],
                                "Matched Food in CSV": item["matched_food"],
                                "Volume (cm³)": item["volume_cm3"],
                                "Density (g/cm³)": f"{density:.2f}",
                                "Weight (g)": f"{item['weight_grams']:.2f} (= {item['volume_cm3']:.2f} × {density:.2f})"
                            })
                        
                        # Display weight calculation table
                        st.table(pd.DataFrame(weight_calc_data))
                        
                        # Now create the nutrition data table
                        for item in nutrition_results:
                            data_dict = {
                                "Food Item": item["food_name"],
                                "Weight (g)": item["weight_grams"],
                                "Calories (kcal/g)": item["calories"],
                                "Protein (g)": item["protein"],
                                "Carbs (g)": item["carbohydrates"],
                                "Fat (g)": item["fat"]
                            }
                            
                            # Add additional nutritional info if available
                            if "fiber" in item:
                                data_dict["Fiber (g)"] = item["fiber"]
                            if "sugars" in item:
                                data_dict["Sugars (g)"] = item["sugars"]
                            if "sodium" in item:
                                data_dict["Sodium (mg)"] = item["sodium"]
                                
                            nutrition_data.append(data_dict)
                        
                        # Display as a table
                        st.table(pd.DataFrame(nutrition_data))
                        
                        # Display total nutrition
                        st.subheader("Total Nutritional Content")
                        
                        # Main nutrients
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Weight", f"{total_nutrition['weight_grams']} g")
                        with col2:
                            st.metric("Calories", f"{total_nutrition['calories']} kcal")
                        with col3:
                            st.metric("Protein", f"{total_nutrition['protein']} g")
                        with col4:
                            st.metric("Carbs", f"{total_nutrition['carbohydrates']} g")
                        with col5:
                            st.metric("Fat", f"{total_nutrition['fat']} g")
                            
                        # Store total calories in session state for health advice tab
                        st.session_state.total_calories = total_nutrition['calories']
                        
                        # Additional nutrients if available
                        if total_nutrition['fiber'] > 0 or total_nutrition['sugars'] > 0 or total_nutrition['sodium'] > 0:
                            st.write("")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Fiber", f"{total_nutrition['fiber']} g")
                            with col2:
                                st.metric("Sugars", f"{total_nutrition['sugars']} g")
                            with col3:
                                st.metric("Sodium", f"{total_nutrition['sodium']} mg")
                        
                        # Add a pie chart for macronutrient breakdown
                        st.subheader("Macronutrient Breakdown")
                        fig, ax = plt.subplots(figsize=(8, 5))
                        labels = ['Protein', 'Carbohydrates', 'Fat']
                        sizes = [total_nutrition['protein'] * 4, total_nutrition['carbohydrates'] * 4, total_nutrition['fat'] * 9]
                        colors = ['#ff9999','#66b3ff','#99ff99']
                        
                        # Only show non-zero values
                        non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
                        if non_zero_indices:
                            ax.pie([sizes[i] for i in non_zero_indices], 
                                  labels=[labels[i] for i in non_zero_indices],
                                  colors=[colors[i] for i in non_zero_indices],
                                  autopct='%1.1f%%', startangle=90)
                            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                            st.pyplot(fig)
                        else:
                            st.info("No macronutrient data available for pie chart")
                        
                        # Add explanation
                        with st.expander("How weight and calories are calculated"):
                            st.write("""
                            ### Weight Calculation
                            The weight of each food item is calculated using the formula:
                            
                            **Weight (g) = Volume (cm³) × Density (g/cm³)**
                            
                            Where:
                            - Volume (cm³) comes from the depth-based volume estimation
                            - Density (g/cm³) is taken directly from the finalcalories.csv file for each specific food
                            
                            ### Calorie Calculation
                            After the weight is calculated, the nutritional values are determined:
                            
                            1. Each nutritional value is calculated by multiplying the weight by the per-gram value from the CSV file
                            2. For example: Calories = Weight (g) × Calories per gram (from CSV)
                            3. The same calculation is applied for protein, carbs, fat, fiber, etc.
                            
                            ### Data Source
                            All nutritional data and density values are taken directly from the finalcalories.csv file.
                            
                            ### Macronutrient Energy Values
                            - Protein: 4 calories per gram
                            - Carbohydrates: 4 calories per gram
                            - Fat: 9 calories per gram
                            """)
                            
                            # Show a sample of the CSV data
                            st.subheader("Sample Data from finalcalories.csv")
                            try:
                                csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finalcalories.csv')
                                csv_sample = pd.read_csv(csv_path).head(5)
                                st.dataframe(csv_sample)                                
                            except Exception as e:
                                st.error(f"Could not load CSV sample: {str(e)}")
                    else:
                        st.warning("Could not find nutritional information for the detected food items")
                        
                except Exception as e:
                    st.error(f"Error during calorie calculation: {str(e)}")
                    st.info("Make sure the food items are correctly detected and volume estimation is completed")
                
        # Health Advice Tab
        with tab4:
            st.subheader("Health Advice")
            
            # Initialize health advisor
            health_advisor = HealthAdvisor()
            
            # Create form for user health data input
            with st.form("health_data_form"):
                st.subheader("Enter Your Health Information")
                
                col1, col2 = st.columns(2)
                with col1:
                    weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.1)
                    height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
                
                with col2:
                    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
                    gender = st.radio("Gender", options=["male", "female"])
                
                # Activity level selection
                activity_options = {
                    "sedentary": "Sedentary (little or no exercise)",
                    "light": "Light (exercise 1-3 days/week)",
                    "moderate": "Moderate (exercise 3-5 days/week)",
                    "active": "Active (exercise 6-7 days/week)",
                    "very_active": "Very Active (hard exercise & physical job)"
                }
                
                activity_level = st.selectbox(
                    "Activity Level", 
                    options=list(activity_options.keys()),
                    format_func=lambda x: activity_options[x],
                    index=2  # Default to moderate
                )
                
                submit_button = st.form_submit_button("Get Health Advice")
            
            if submit_button:
                # Calculate BMI
                bmi = health_advisor.calculator.calculate_bmi(weight_kg, height_cm)
                bmi_category = health_advisor.calculator.get_bmi_category(bmi)
                
                # Calculate daily calorie needs
                daily_calorie_needs = health_advisor.calculator.calculate_daily_calorie_needs(
                    weight_kg, height_cm, age, gender, activity_level
                )
                
                # Calculate meal distribution
                meal_distribution = health_advisor.calculator.calculate_meal_calorie_distribution(daily_calorie_needs)
                
                # Display BMI information
                st.subheader("Your BMI Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("BMI", f"{bmi:.1f}")
                    st.info(f"Category: {bmi_category['category'].title()}")
                
                with col2:
                    st.write("**BMI Categories:**")
                    st.write("- Underweight: < 18.5")
                    st.write("- Normal weight: 18.5–24.9")
                    st.write("- Overweight: 25–29.9")
                    st.write("- Obesity: ≥ 30")
                
                st.write(f"**Health Advice:** {bmi_category['advice']}")
                
                # Display calorie needs
                st.subheader("Your Daily Calorie Needs")
                st.metric("Estimated Daily Calorie Needs", f"{round(daily_calorie_needs)} kcal")
                
                # Display meal distribution
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Breakfast", f"{meal_distribution['breakfast']} kcal")
                with col2:
                    st.metric("Lunch", f"{meal_distribution['lunch']} kcal")
                with col3:
                    st.metric("Dinner", f"{meal_distribution['dinner']} kcal")
                with col4:
                    st.metric("Snacks", f"{meal_distribution['snacks']} kcal")
                
                # If we have detected food calories, analyze them
                if 'total_calories' in st.session_state:
                    total_calories = st.session_state.total_calories
                    
                    st.subheader("Current Meal Analysis")
                    meal_analysis = health_advisor.calculator.analyze_meal_calories(total_calories, daily_calorie_needs)
                    
                    # Display meal analysis
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Detected Meal Calories", f"{total_calories} kcal")
                    with col2:
                        st.metric("Recommended Meal Calories", f"{meal_analysis['recommended_meal_calories']} kcal")
                    
                    # Show status with appropriate color
                    if meal_analysis['status'] == 'above':
                        st.warning(meal_analysis['message'])
                    elif meal_analysis['status'] == 'below':
                        st.info(meal_analysis['message'])
                    else:
                        st.success(meal_analysis['message'])
                    
                    # Provide specific advice based on BMI category and meal calories
                    st.subheader("Personalized Health Recommendations")
                    
                    if bmi_category['category'] == 'overweight' or bmi_category['category'] == 'obese':
                        if meal_analysis['status'] == 'above':
                            st.warning("Your meal contains more calories than recommended and your BMI indicates you may be overweight.")
                            st.write("**Recommendations:**")
                            st.write("- Consider reducing portion sizes")
                            st.write("- Choose lower-calorie alternatives")
                            st.write("- Increase intake of vegetables and fiber-rich foods")
                            st.write("- Aim for a daily calorie deficit of 500-750 kcal for healthy weight loss")
                        elif meal_analysis['status'] == 'within':
                            st.success("Your meal is within the recommended calorie range, but your BMI indicates you may be overweight.")
                            st.write("**Recommendations:**")
                            st.write("- Continue monitoring portion sizes")
                            st.write("- Increase physical activity")
                            st.write("- Focus on nutrient-dense foods")
                            st.write("- Consider a slight calorie deficit for gradual weight loss")
                        else:  # below
                            st.success("Your meal contains fewer calories than recommended, which may help with weight management.")
                            st.write("**Recommendations:**")
                            st.write("- Ensure you're getting adequate nutrition despite lower calories")
                            st.write("- Focus on protein-rich foods to maintain muscle mass")
                            st.write("- Stay hydrated throughout the day")
                            st.write("- Maintain this calorie level if it's sustainable for you")
                    
                    elif bmi_category['category'] == 'underweight':
                        if meal_analysis['status'] == 'below':
                            st.warning("Your meal contains fewer calories than recommended and your BMI indicates you may be underweight.")
                            st.write("**Recommendations:**")
                            st.write("- Consider increasing portion sizes")
                            st.write("- Add nutrient and calorie-dense foods to your diet")
                            st.write("- Include healthy fats like nuts, avocados, and olive oil")
                            st.write("- Aim for a daily calorie surplus of 300-500 kcal for healthy weight gain")
                        else:  # within or above
                            st.success("Your meal contains adequate or higher calories, which is good for your underweight status.")
                            st.write("**Recommendations:**")
                            st.write("- Continue with calorie-dense, nutritious meals")
                            st.write("- Consider strength training to build muscle mass")
                            st.write("- Eat frequent meals and snacks throughout the day")
                            st.write("- Monitor your weight gain to ensure it's healthy")
                    
                    else:  # normal weight
                        if meal_analysis['status'] == 'within':
                            st.success("Your meal is within the recommended calorie range and your BMI is in the healthy range.")
                            st.write("**Recommendations:**")
                            st.write("- Maintain your current eating patterns")
                            st.write("- Continue with regular physical activity")
                            st.write("- Focus on a balanced diet with a variety of foods")
                            st.write("- Regular health check-ups to monitor your health status")
                        elif meal_analysis['status'] == 'above':
                            st.info("Your meal contains more calories than recommended, although your BMI is in the healthy range.")
                            st.write("**Recommendations:**")
                            st.write("- Monitor your overall daily calorie intake")
                            st.write("- Balance higher-calorie meals with more physical activity")
                            st.write("- Ensure you're not regularly exceeding your daily calorie needs")
                            st.write("- Focus on nutrient-dense rather than calorie-dense foods")
                        else:  # below
                            st.info("Your meal contains fewer calories than recommended, although your BMI is in the healthy range.")
                            st.write("**Recommendations:**")
                            st.write("- Ensure you're meeting your nutritional needs across all meals")
                            st.write("- Consider adding more calories in other meals if this is a pattern")
                            st.write("- Monitor your weight to ensure it remains stable")
                            st.write("- Focus on balanced nutrition rather than just calorie counting")
                else:
                    st.info("Complete the Calorie Calculation tab to get personalized meal analysis based on detected food.")
                    
                # Display a chart comparing current meal to daily needs
                if 'total_calories' in st.session_state:
                    st.subheader("Calorie Comparison")
                    
                    # Create data for the chart
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    categories = ['Current Meal', 'Recommended Meal', 'Daily Needs']
                    values = [total_calories, meal_analysis['recommended_meal_calories'], round(daily_calorie_needs)]
                    
                    # Create bar chart
                    bars = ax.bar(categories, values, color=['#ff9999', '#66b3ff', '#99ff99'])
                    
                    # Add data labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                                f'{int(height)} kcal', ha='center', va='bottom')
                    
                    # Customize chart
                    ax.set_ylabel('Calories (kcal)')
                    ax.set_title('Calorie Comparison')
                    plt.tight_layout()
                    
                    # Display the chart
                    st.pyplot(fig)
                
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

def main():
    st.title("Food Detection and Nutrition Analysis System")
    
    # Initialize session state for storing data between tabs
    if 'food_volumes' not in st.session_state:
        st.session_state.food_volumes = []
    
    try:
        detector = ViewDetector()  # Changed from FoodDetector to ViewDetector
        
        uploaded_file = st.file_uploader(
            "Upload an image of food", 
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file:
            process_image(uploaded_file, detector)
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Make sure the model file 'final_yolov8.pt' is in the correct location")

if __name__ == "__main__":
    main()