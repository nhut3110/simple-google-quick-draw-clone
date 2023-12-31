from uuid import uuid4
from os import remove
from PIL import Image, ImageDraw
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from starlette.responses import FileResponse
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

class ImageData(BaseModel):
    strokes: list
    box: list

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transform")
async def transform(image_data: ImageData):
    img = transform_img(image_data.strokes, image_data.box)
    
    # Use BytesIO to keep the image in memory
    img_data = BytesIO()
    img.save(img_data, format='PNG')
    
    # Go to the start of the BytesIO stream
    img_data.seek(0)

    return StreamingResponse(img_data, media_type="image/png")


app.mount("/", StaticFiles(directory="static", html=True), name="static")


def transform_img(strokes, box):
    # Calc cropped image size
    width = box[2] - box[0]
    height = box[3] - box[1]

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    image_draw = ImageDraw.Draw(image)

    for stroke in strokes:
        positions = []
        for i in range(0, len(stroke[0])):
            positions.append((stroke[0][i], stroke[1][i]))
        image_draw.line(positions, fill=(0, 0, 0), width=3)

    return image.resize(size=(28, 28))
