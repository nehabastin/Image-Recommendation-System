import nest_asyncio
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
import openai
from pyunsplash import PyUnsplash

import dotenv
from openai import OpenAI

#load the environment variables
OPENAI_API = dotenv.get('OPENAI_API')
UNSPLASH_API = dotenv.get('UNSPLASH_API')

# Patch the event loop
nest_asyncio.apply()

app = FastAPI()
client = OpenAI( api_key=OPENAI_API,
)
# Set your OpenAI AP
unsplash_api_key = UNSPLASH_API

pu = PyUnsplash(api_key=unsplash_api_key)

def analyze_text_for_keywords(content: str):
    """
    Analyzes the provided text to extract keywords using the OpenAI API.
    """
    client = OpenAI(
    # This is the default and can be omitted
    api_key=OPENAI_API,
    )
    prompt = f"Analyze the following text and generate a list of keywords: {content}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a keyword extraction assistant,  Understand the context and its supposed to be used to find sutable images and videos add relevant additional keywords based on the scenario. Order the keywords based on relevance of the content, List the keyword only  as the response"},
            {"role": "user", "content": prompt}
        ]
    )
    keywords = response.choices[0].message
    keywords = keywords.content
    keywords = keywords.split(",")

    return keywords

def generate_ai_image_prompt(content: str):
    """
    Generates a detailed prompt for AI image generation based on the text content.
    """
    prompt = f"Create a detailed and specific image generation prompt for the following content: {content}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a AI Image Prompt Generator. Provide Prompts to generate image in DALLE using Photo Realistic Images with real world lighting , ,  must be under 200 tokens in total only return just the prompt and no other text"},
            {"role": "user", "content": prompt}
        ]
    )
    imageprompt = response.choices[0].message
    imageprompt = imageprompt.content
    return imageprompt

def generate_ai_image(PROMPT: str):
    """
    Generates images using the OpenAI API based on the provided prompt.
    """
    client = OpenAI(
        # This is the default and can be omitted
        api_key=OPENAI_API,
        )
    response = client.images.generate(
        prompt=PROMPT,
        n=2,
        size="512x512"
    )
    url = response.data[0].url
    return url

def fetch_unsplash_images(keywords: str):
    """
    Fetches images from Unsplash based on the provided keywords using PyUnsplash.
    """
    #loop through the list and get the images
    search = pu.search(type_="photos", query=keywords, per_page=1)
    images = [photo.link_download for photo in search.entries]
    return images[0]

@app.get("/status")
async def status_check():
    return JSONResponse(content={"status": "API is running"})

@app.get("/recommend_images")
async def recommend_images(query: Optional[str] = Query(None, description="Query string for image recommendation"),
                           use_ai: Optional[bool] = Query(False, description="Use AI to generate images")):
    if query:
      if use_ai:
            PROMPT = generate_ai_image_prompt(query)
            IMAGE_URL = generate_ai_image(PROMPT)
            #return JSONResponse(content={"url": IMAGE_URL})
            return JSONResponse(IMAGE_URL)
      else:
            keywords = analyze_text_for_keywords(query)
            images = fetch_unsplash_images(keywords)
            return JSONResponse(images)
            # return JSONResponse(content={"keywords": keywords, "images": images})
    else:
        image_path = "/content/image1.png"  # Replace with logic to select the appropriate image
        return FileResponse(image_path, media_type="image/png")




# Run the Uvicorn server
uvicorn.run(app, host="0.0.0.0", port=8000)