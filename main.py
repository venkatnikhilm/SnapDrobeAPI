import os
import base64
from google import genai
from google.genai import types
from uagents import Agent, Context, Model
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
