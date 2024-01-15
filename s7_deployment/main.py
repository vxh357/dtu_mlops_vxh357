from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from typing import Optional
from enum import Enum
from http import HTTPStatus
import re
from pydantic import BaseModel
import cv2

app = FastAPI()

@app.get("/hello_world")
def read_root():
    """Health check endpoint."""
    return {"Hello": "World"}

class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

class EmailCheckRequest(BaseModel):
    email: str
    domain_match: str

@app.get("/query_items/{item_id}")
def read_item(item_id: int):
    """
    Get an item by item_id.

    Args:
        item_id (int): The item's unique identifier.

    Returns:
        dict: A JSON response with the item_id.
    """
    return {"item_id": item_id}

@app.get("/")
def root():
    """
    Root endpoint for health check.

    Returns:
        dict: A JSON response with a health check message.
    """
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK.value,
    }
    return response

database = {'username': [], 'password': []}

@app.post("/login/")
def login(username: str, password: str):
    """
    Log in a user with the provided username and password.

    Args:
        username (str): The user's username.
        password (str): The user's password.

    Returns:
        str: A confirmation message indicating a successful login.
    """
    username_db = database['username']
    password_db = database['password']
    
    if username not in username_db and password not in password_db:
        with open('database.csv', "a") as file:
            file.write(f"{username}, {password}\n")
        username_db.append(username)
        password_db.append(password)
    
    return "Login saved"

@app.post("/text_model/")
def contains_email(data: EmailCheckRequest):
    """
    Check if the provided email and domain match a specific pattern.

    Args:
        data (EmailCheckRequest): The JSON object containing email and domain to be checked.

    Returns:
        dict: A JSON response with information about the input email and whether it matches the domain.
    """
    email = data.email
    domain_match = data.domain_match

    # Define regex patterns for gmail and hotmail domains
    gmail_pattern = r'\b[A-Za-z0-9._%+-]+@gmail\.com\b'
    hotmail_pattern = r'\b[A-Za-z0-9._%+-]+@hotmail\.com\b'

    if domain_match == "gmail" and re.fullmatch(gmail_pattern, email):
        match = True
    elif domain_match == "hotmail" and re.fullmatch(hotmail_pattern, email):
        match = True
    else:
        match = False

    response = {
        "input_email": email,
        "domain_match": domain_match,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK.value,
        "matches_domain": match,
    }

    return response

@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: Optional[int] = 28, w: Optional[int] = 28):
    with open('image.jpg', 'wb') as image:
        content = await data.read()
        image.write(content)

    # Read the uploaded image using OpenCV
    img = cv2.imread("image.jpg")

    # Resize the image using the provided height (h) and width (w)
    if h > 0 and w > 0:
        img = cv2.resize(img, (h, w))

    # Save the resized image back to the file
    cv2.imwrite("image_resize.jpg", img)

    # Return the resized image as a FileResponse
    return FileResponse("image_resize.jpg", media_type="image/jpeg")