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

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
dynamo_client = boto3.client(
    "dynamodb",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

class Article(BaseModel):
    main_category: str
    sub_category: str
    primary_color: str
    secondary_color: str
    other_colors: list[str]
    pattern: str
    description: str
    gender: str
    approximate_age_of_wearer: str
    occasion: str
    season: str
    style: str
    ways_to_wear: list[str]
    image_b64: str

class GeminiArticleResponse(BaseModel):
    article: Article
    return_response: str

class ExplainRequest(Model):
    image_b64: str

class ExplainResponse(Model):
    mind_map_image_url: str
    transcription: str

agent = Agent(
    name="explanation_agent",
    seed=os.getenv("AGENT_SEED"),
    port=8001,
    endpoint=["http://0.0.0.0:8001/submit"],
)

@agent.on_rest_post("/add_image", ExplainRequest, ExplainResponse)
async def handle_add_image(ctx: Context, req: ExplainRequest) -> ExplainResponse:
    img_bytes = base64.b64decode(req.image_b64)
    tmp_path = "/tmp/cap.jpg"
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)

    uploaded = client.files.upload(file=tmp_path)
    mind = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_text(text="Based on the input image, extract all the relevant information and a sassy response to say that the image has been added to the database (Basically, remembered)."),
            types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
        ],
        config={
            "response_type": "application/json",
            "response_schema": GeminiArticleResponse
        }
    )
    response = json.loads(mind.text)
    data = response["article"]
    data["id"] = str(uuid4())

    dynamo_client.put_item(
        TableName=os.getenv("DYNAMODB_TABLE_NAME"),
        Item=data
    )

    return response["return_response"]


@agent.on_rest_post("/explain", ExplainRequest, ExplainResponse)
async def handle_explain(ctx: Context, req: ExplainRequest) -> ExplainResponse:
    img_bytes = base64.b64decode(req.image_b64)
    tmp_path = "/tmp/cap.jpg"
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)

    uploaded = client.files.upload(file=tmp_path)
    mind = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=[
            types.Part.from_text(text="Create an image that visualizes the story from the text in the input image"),
            types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
        ],
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )

    candidates = mind.candidates or []
    if not candidates:
        raise RuntimeError("No candidates returned from image generation")
    parts = candidates[0].content.parts or []
    img_part = next((p for p in parts if p.inline_data), None)
    if img_part is None:
        raise RuntimeError(f"No inline image data found; available parts: {parts!r}")

    mime = img_part.inline_data.mime_type
    img_b64 = base64.b64encode(img_part.inline_data.data).decode()
    img = base64.b64decode(img_b64)
    mind_url = f"data:{mime};base64,{img_b64}"
    tmp_path = "mind_map.jpg"
    with open(tmp_path, "wb") as f:
        f.write(img)

    summ = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[
            types.Part.from_text(text="Summarize this image in simple language."),
            types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
        ],
    )

    return ExplainResponse(
        mind_map_image_url=mind_url,
        transcription=summ.text,
    )

if __name__ == "__main__":
    agent.run()
