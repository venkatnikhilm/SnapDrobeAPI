# agent.py
import os
import base64
from google import genai
from google.genai import types
from uagents import Agent, Context, Model
from dotenv import load_dotenv

load_dotenv()   # Load environment variables from .env file


# — Configure Gemini SDK
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# client = genai.Client()
# filepath: /Users/venkatnikhilm/Desktop/Projects/Specatcles_Project_LAHacks/main.py
# Replace the configure line
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Pass the API key directly to the client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# — Define your request/response schemas
class ExplainRequest(Model):
    image_b64: str

class ExplainResponse(Model):
    mind_map_image_url: str
    transcription: str

# — Create the agent and register a REST POST endpoint
agent = Agent(
    name="explanation_agent",
    seed=os.getenv("AGENT_SEED"),
    port=8001,
    endpoint=["http://0.0.0.0:8001/submit"],
)

@agent.on_rest_post("/explain", ExplainRequest, ExplainResponse)
async def handle_explain(ctx: Context, req: ExplainRequest) -> ExplainResponse:
    # 1) Decode image
    img_bytes = base64.b64decode(req.image_b64)
    tmp = "/tmp/cap.jpg"
    with open(tmp, "wb") as f:
        f.write(img_bytes)

    # 2) Upload & generate mind-map image
    uploaded = client.files.upload(file=tmp)
    mind = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=[
            types.Part.from_text("Create a mind-map illustrating the key concepts."),
            types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
        ],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"])
    )
    # extract inline image
    part = next(p for p in mind.candidates[0].content.parts if p.inline_data)
    mime = part.inline_data.content_type
    img_b64 = base64.b64encode(part.inline_data.data).decode()
    mind_url = f"data:{mime};base64,{img_b64}"

    # 3) Generate transcription
    summ = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[
            types.Part.from_text("Summarize this image in simple language."),
            types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
        ],
    )

    return ExplainResponse(
        mind_map_image_url=mind_url,
        transcription=summ.text
    )

if __name__ == "__main__":
    agent.run()