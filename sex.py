# agent.py
import os
import base64
from google import genai
from google.genai import types
from uagents import Agent, Context, Model
from dotenv import load_dotenv

load_dotenv()  # loads GEMINI_API_KEY and AGENT_SEED

# Initialize the Gemini SDK client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Request/response schemas
class ExplainRequest(Model):
    image_b64: str

class ExplainResponse(Model):
    mind_map_image_url: str
    transcription: str

# Create the uAgent
agent = Agent(
    name="explanation_agent",
    seed=os.getenv("AGENT_SEED"),
    port=8001,
    endpoint=["http://0.0.0.0:8001/submit"],
)

@agent.on_rest_post("/explain", ExplainRequest, ExplainResponse)
async def handle_explain(ctx: Context, req: ExplainRequest) -> ExplainResponse:
    # 1) Decode the incoming image
    img_bytes = base64.b64decode(req.image_b64)
    tmp_path = "capture.jpg"
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)

    # 2) Upload to Gemini
    uploaded = client.files.upload(file=tmp_path)  # returns a File object  [oai_citation:0‡Google APIs](https://googleapis.github.io/python-genai/?utm_source=chatgpt.com)

    # 3) Generate a mind-map image
    mind_resp = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=[
            "Create a mind-map illustrating the key concepts in this image.",
            uploaded
        ],
        # config=types.GenerateContentConfig(response_modalities=["IMAGE"])
    )
    # Extract the inline image bytes
    part = next(p for p in mind_resp.candidates[0].content.parts if p.inline_data)
    mime = part.inline_data.content_type
    b64_img = base64.b64encode(part.inline_data.data).decode()
    mind_map_url = f"data:{mime};base64,{b64_img}"
    img_bytes = base64.b64decode(b64_img)
    tmp_path = "resp.jpg"
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)

    # 4) Generate a text summary
    sum_resp = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[
            "Summarize what is shown in this image in simple, conversational language.",
            uploaded
        ],
    )

    # 5) Return both
    return ExplainResponse(
        mind_map_image_url=mind_map_url,
        transcription=sum_resp.text
    )

if __name__ == "__main__":
    agent.run()