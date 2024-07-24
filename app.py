from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline
import torch
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from functools import lru_cache
import logging

# Log Config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting model warmup...")
    try:
        dummy_image = Image.new('RGB', (512, 512))
        dummy_mask = Image.new('L', (512, 512))
        await asyncio.to_thread(warmup_model, dummy_image, dummy_mask)
        logger.info("Model warmup completed successfully.")
    except Exception as e:
        logger.error(f"Error during model warmup: {str(e)}")
    yield
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16).to(device)

PROMPT = "a photo of an object, inpainting only inside the mask with a similar type of object, high resolution, realistic details, seamless blending"

@lru_cache(maxsize=1)
def get_optimized_prompt_embeds(prompt):
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = pipe.text_encoder(text_input_ids.to(device))[0]
    return prompt_embeds

@torch.inference_mode()
def generate_image(image, mask_image):
    prompt_embeds = get_optimized_prompt_embeds(PROMPT)
    return pipe(
        prompt_embeds=prompt_embeds,
        image=image,
        mask_image=mask_image,
        num_inference_steps=15,
        strength=1,
        guidance_scale=20
    ).images[0]

def warmup_model(dummy_image, dummy_mask):
    logger.info("Starting generate_image in warmup...")
    try:
        _ = generate_image(dummy_image, dummy_mask)
        logger.info("generate_image completed successfully in warmup.")
    except Exception as e:
        logger.error(f"Error in generate_image during warmup: {str(e)}")
        raise

    
async def process_image(image_data, mask_coords):
    image = await asyncio.to_thread(Image.open, BytesIO(image_data))
    image = await asyncio.to_thread(image.convert, 'RGB')
    image = await asyncio.to_thread(image.resize, (512, 512), Image.LANCZOS)
    
    mask_image = await asyncio.to_thread(Image.new, 'L', (512, 512), 0)
    draw = ImageDraw.Draw(mask_image)
    await asyncio.to_thread(draw.rectangle, mask_coords, fill=255)
    
    return image, mask_image

@app.post('/inpaint')
async def inpaint_image(
    image: UploadFile = File(...),
    maskX1: int = Form(...),
    maskY1: int = Form(...),
    maskX2: int = Form(...),
    maskY2: int = Form(...)
):
    try:
        contents = await image.read()
        image, mask_image = await process_image(contents, [maskX1, maskY1, maskX2, maskY2])
        
        inpainted_image = await asyncio.to_thread(generate_image, image, mask_image)
        
        img_byte_arr = BytesIO()
        await asyncio.to_thread(inpainted_image.save, img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type='image/png')
    
    except Exception as e:
        logger.error(f"Error in inpaint_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=5000)