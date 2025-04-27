import os
import base64
from google import genai
from google.genai import types
from uagents import Agent, Context, Model
from dotenv import load_dotenv
import json
from pydantic import BaseModel
import boto3
from uuid import uuid4
from boto3.dynamodb.types import TypeSerializer, TypeDeserializer
from typing import Literal
import requests
from PIL import Image
from io import BytesIO
import base64
# from elevenlabs.client import ElevenLabs
# from elevenlabs import play

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
dynamo_client = boto3.client(
    "dynamodb",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
# tts = ElevenLabs(
#   api_key=os.getenv("ELEVENLABS_API_KEY"),
# )

class Article(BaseModel):
    main_category: Literal["Topwear", "Bottomwear", "Innerwear", "Footwear", "Accessories"]
    sub_category: Literal["T-shirt", "Shirt", "Pants", "Shorts", "Skirt", "Dress", "Shoes", "Socks", "Belt", "Hat", "Scarf", "Jacket", "Coat", "Sweater", "Gloves", "Bag", "Jewelry", "Watch", "Sunglasses", "Umbrella"]
    primary_color: str
    secondary_color: str
    other_colors: list[str]
    pattern: Literal["Solid", "Striped", "Checked", "Floral", "Polka Dot", "Geometric", "Animal Print", "Graphic", "Textured", "Plaid"]
    description: str
    gender: Literal["Male", "Female", "Unisex"]
    age_group: Literal["Child", "Teenager", "Young Adult", "Adult", "Senior"]
    occasion: Literal["Casual", "Formal", "Party", "Sports", "Beach", "Travel", "Wedding", "Office", "Outdoor", "Gym"]
    weather: Literal["Sunny", "Rainy", "Snowy", "Windy", "Cloudy", "Foggy", "Stormy", "Hot", "Cold"]
    style_description: str
    ways_to_wear: list[str]

class GeminiArticleResponse(BaseModel):
    article: Article
    return_response: str

class AddImageRequest(Model):
    image_b64: str

class AddImageResponse(Model):
    return_response: str
    # audio_blob: bytes

class AskRequest(Model):
    input_prompt: str

class AskResponse(Model):
    return_prompt: str
    image_b64: str
    # articles: list[dict]

class WardrobeRequest(Model):
    pass

class WardrobeResponse(Model):
    articles: list

class RootResponse(BaseModel):
    message: str
    endpoints: list[dict]



port = int(os.getenv("PORT", 8000))  # fallback for local dev
agent = Agent(
    name="explanation_agent",
    seed=os.getenv("AGENT_SEED"),
    port=port,
    endpoint=[f"http://0.0.0.0:{port}/submit"],
)

@agent.on_rest_get("/", RootResponse)
async def handle_root(ctx: Context) -> RootResponse:
    return RootResponse(
        message="Welcome to the Snapdrobe API!",
        endpoints=[
            {"method": "POST", "path": "/add_image", "description": "Add an image to the database."},
            {"method": "POST", "path": "/ask", "description": "Ask for outfit recommendations based on input."},
        ]
    )

@agent.on_rest_post("/add_image", AddImageRequest, AddImageResponse)
async def handle_add_image(ctx: Context, req: AddImageRequest) -> AddImageResponse:
    img_bytes = base64.b64decode(req.image_b64)
    prompt = f"""
        Based on the input image, extract all the relevant information and a
        sassy response to say that the image has been added to the database (Basically, remembered).
        Return the output in the following schema:
        {json.dumps(GeminiArticleResponse.model_json_schema(), indent=2)}

        Remember - 
            - The colours should always be in hex format."""
    mind = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=prompt),
        ],
    )
    output = mind.text.replace("```json", "").replace("```", "")
    if not output:
        raise ValueError("The response from generate_content is empty or invalid.")
    try:
        response = json.loads(output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e}") from e
    data = response["article"]
    data["id"] = str(uuid4())

    # Serialize the data for DynamoDB
    serializer = TypeSerializer()
    dynamo_item = {k: serializer.serialize(v) for k, v in data.items()}

    dynamo_client.put_item(
        TableName=os.getenv("DYNAMODB_TABLE_NAME"),
        Item=dynamo_item
    )

    # audio = tts.text_to_speech.convert(
    #     text=response["return_response"],
    #     voice_id="56AoDkrOh6qfVPDXZ7Pt",
    #     model_id="eleven_multilingual_v2",
    #     output_format="mp3_44100_128",
    # )

    return AddImageResponse(
        return_response=response["return_response"],
        # audio_blob=audio,
    )

# @agent.on_rest_get("/wardrobe", WardrobeRequest, WardrobeResponse)
# async def handle_wardrobe(ctx: Context, req: WardrobeRequest) -> WardrobeResponse:
#     response = dynamo_client.scan(
#         TableName=os.getenv("DYNAMODB_TABLE_NAME"),
#     )
#     items = response.get("Items", [])
#     return WardrobeResponse(
#         articles=items
#     )    

@agent.on_rest_post("/ask", AskRequest, AskResponse)
async def handle_ask(ctx: Context, req: AskRequest) -> AskResponse:
    input_prompt = req.input_prompt
    weather_api_key = os.getenv("OPENWEATHER_API_KEY")
    location = "Los Angeles,US"
    # weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_api_key}&units=metric"
    weather_url = f"https://api.openweathermap.org/data/3.0/onecall?lat=34.0549&lon=-118.2426&exclude=minutely,hourly,alerts&units=metric&appid=4244499cf74b39bae97df3fa9189abbf" 
    gender = "Female"

    weather_response = requests.get(weather_url)
    if weather_response.status_code != 200:
        raise Exception(f"Failed to fetch weather data: {weather_response.status_code} {weather_response.text}")
    
    weather_data = weather_response.json()
    # weather_description = weather_data["weather"][0]["description"]
    # temperature = {weather_data.temperature}

    

    response = dynamo_client.scan(
        TableName=os.getenv("DYNAMODB_TABLE_NAME"),
    )
    deserializer = TypeDeserializer()
    items = list(map(lambda item: {k: deserializer.deserialize(v) for (k, v) in item.items()}, response.get("Items", [])))
    # items_schema = 
    combined_prompt = f"""
    Input Prompt: {input_prompt}
    
    Current Location: {location}
    Current Weather: {weather_data}
    Gender: {gender}
    
    Using this information, select an outfit consisting of articles that are appropriate for the given input and weather conditions.

    Wardrobe Database:
    {json.dumps(items, indent=2)}
    
    Return the output in the following schema:

    {{
        "articles": [{json.dumps(Article.model_json_schema(), indent=2)}],
        "return_prompt": "text describing the outfit in a sassy/cool way"
    }}
    """
    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=combined_prompt,
    )
    data = json.loads(response.text.replace("```json", "").replace("```", ""))
    articles = data["articles"]
    return_prompt = data["return_prompt"]
    # --- Image Generation Step (using the correct experimental model) ---

    full_image_prompt = f"""Generate a realistic image of a person wearing this outfit. Match the styles, fabrics, colors (in HEX), patterns, and descriptions precisely. The image should show the complete head to torse of the person wearing this outfit Clothing Articles:{json.dumps(articles, indent=2)}"""

    image_generation_response = client.models.generate_content(
    model="gemini-2.0-flash-exp-image-generation",
    contents=full_image_prompt,
    config=types.GenerateContentConfig(
    response_modalities=['TEXT', 'IMAGE']
    )    )

        # Extract the image data from the part
    for part in image_generation_response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO((part.inline_data.data)))
            # image.save('gemini-native-image.png')
            image_b64 = base64.b64encode(image.tobytes()).decode('utf-8')



    # --- End of Image Generation Step ---

    return AskResponse(
        return_prompt=return_prompt,
        image_b64=image_b64
        # You might still want to return the articles data
        # articles=articles,
        # outfit_image_b64=image_b64 # Return the generated image (if successful)
    )



if __name__ == "__main__":
    agent.run()
