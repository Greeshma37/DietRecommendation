from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import requests
import re
import os

# Initialize OpenAI API key and endpoint
openai.api_key = os.environ.get("api_key")  # Replace with your API key
endpoint = os.environ.get("endpoint")    # Replace with your Azure OpenAI endpoint URL
deployment_name = os.environ.get("deployment_name")    # Replace with your deployment name

# Define FastAPI app
app = FastAPI()

# Input JSON structure
class UserInfo(BaseModel):
    age: int
    weight: float
    height: float
    symptoms: str
    medical_conditions: str
    activity_level: str
    dietary_preferences: str
    preferred_cuisine: str
    sleep_pattern: str
    stress_level: str

# Function to fetch nutrient requirements and food recommendations
def get_nutrient_and_food_recommendations(user_info):
    prompt = f"""
    Based on the following user profile, estimate the daily nutrient requirements and recommend specific foods with their nutritional values. Make sure you take the user's symptoms into account and answer accordingly.
    The recommendations should consider the user's preferences and cultural dietary habits. Provide the response in two parts:
    1. Daily Nutrient Requirements as structured text.
    2. Food Suggestions with their nutritional values as structured text.

    Here is the user profile:
    - Age: {user_info.age}
    - Weight: {user_info.weight}
    - Height: {user_info.height}
    - Symptoms: {user_info.symptoms}
    - Medical Conditions: {user_info.medical_conditions}
    - Activity Level: {user_info.activity_level}
    - Dietary Preferences: {user_info.dietary_preferences}
    - Preferred Cuisine: {user_info.preferred_cuisine}
    - Sleep Pattern: {user_info.sleep_pattern}
    - Stress Level: {user_info.stress_level}

    Respond in the following format:
    1. Daily Nutrient Requirements:
    Nutrient: Value (Unit)

    2. Food Suggestions:
    Food Item: Nutrient1: Value1 (Unit), Nutrient2: Value2 (Unit), ...
    """

    headers = {
        "Content-Type": "application/json",
        "api-key": openai.api_key,
    }
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant providing dietary recommendations."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1500,
        "temperature": 0.7,
    }

    response = requests.post(
        f"{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version=2023-06-01-preview",
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        response_text = response.json()["choices"][0]["message"]["content"]
        return {"recommendations": response_text}
    else:
        raise HTTPException(status_code=500, detail=f"GPT-4 API Error: {response.status_code}, {response.text}")




# Function to translate and summarize the response
def translate_and_summarize(nutrient_requirements, target_language):
    prompt = f"""
    Translate and summarize the following information into {target_language}:
    {nutrient_requirements}

    """

    headers = {
        "Content-Type": "application/json",
        "api-key": openai.api_key,
    }
    data = {
        "messages": [
            {"role": "system", "content": f"You are a helpful assistant translating into {target_language}."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2000,
        "temperature": 0.7,
    }

    response = requests.post(
        f"{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version=2023-06-01-preview",
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

# API endpoint to get recommendations
@app.post("/get-recommendations")
def get_recommendations(user_info: UserInfo):
    try:
        nutrient_requirements = get_nutrient_and_food_recommendations(user_info)

        if not nutrient_requirements:
            raise HTTPException(status_code=500, detail="Failed to retrieve recommendations.")

        summary_in_hindi = translate_and_summarize(nutrient_requirements, target_language="Hindi")

        return {
            "daily_nutrient_requirements": nutrient_requirements,
            "summary_in_hindi": summary_in_hindi
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
