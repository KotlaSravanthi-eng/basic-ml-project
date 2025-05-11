import gradio as gr
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Prediction function
def predict_performance(gender, ethnicity, parental_level_of_education, lunch,
                        test_preparation_course, reading_score, writing_score):
    
    # Create data object
    data = CustomData(
        gender=gender,
        race_ethnicity=ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )
    
    # Convert to dataframe and predict
    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()
    result = predict_pipeline.predict(pred_df)
    return f"Predicted Math Score: {float(result[0])}"

# Create Gradio interface
demo = gr.Interface(
    fn=predict_performance,
    inputs=[
        gr.Dropdown(["male", "female"], label="Gender"),
        gr.Dropdown(["group A", "group B", "group C", "group D", "group E"], label="Race/Ethnicity"),
        gr.Dropdown(["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"], label="Parental Level of Education"),
        gr.Dropdown(["standard", "free/reduced"], label="Lunch Type"),
        gr.Dropdown(["none", "completed"], label="Test Preparation Course"),
        gr.Slider(0, 100, step=1, label="Reading Score"),
        gr.Slider(0, 100, step=1, label="Writing Score"),
    ],
    outputs="text",
    title="Student Performance Predictor",
    description="Predict the math score of a student based on background and test scores."
)

demo.launch(share = True, ssr_mode= False)
