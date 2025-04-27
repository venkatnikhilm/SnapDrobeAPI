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
from elevenlabs.client import ElevenLabs
from elevenlabs import play

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
dynamo_client = boto3.client(
    "dynamodb",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
tts = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

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
    articles: list[dict]

class WardrobeRequest(Model):
    pass

class WardrobeResponse(Model):
    articles: list

agent = Agent(
    name="explanation_agent",
    seed=os.getenv("AGENT_SEED"),
    port=8002,
    endpoint=["http://0.0.0.0:8002/submit"],
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

    audio = tts.text_to_speech.convert(
        text=response["return_response"],
        voice_id="56AoDkrOh6qfVPDXZ7Pt",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

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
    response = dynamo_client.scan(
        TableName=os.getenv("DYNAMODB_TABLE_NAME"),
    )
    deserializer = TypeDeserializer()
    items = list(map(lambda item: {k: deserializer.deserialize(v) for (k, v) in item.items()}, response.get("Items", [])))
    # items_schema = 
    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=f"Using the following input and the database return an outfit consisting of a bunch of articles that are appropriate to wear given the conditions mentioned in the input prompt.\n\nInput Prompt - \n{input_prompt}\n\nDatabase - \n{json.dumps(items, indent=2)}\n\nReturn the output in the following schema: \n\n\n{{articles: [{json.dumps(Article.model_json_schema(), indent=2)}], return_prompt: str}}",
    )
    data = json.loads(response.text.replace("```json", "").replace("```", ""))
    return AskResponse(
        return_prompt=data["return_prompt"],
        articles=data["articles"]
    )

if __name__ == "__main__":
    agent.run()
