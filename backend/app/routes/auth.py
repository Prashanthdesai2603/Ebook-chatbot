from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# In-memory user store for simulation
USERS = {
    "admin": "admin123"
}

class LoginRequest(BaseModel):
    username: str
    password: str

class SignupRequest(BaseModel):
    username: str
    password: str
    email: str

class AuthResponse(BaseModel):
    success: bool
    message: str
    token: Optional[str] = None

@router.post("/signup", response_model=AuthResponse)
async def signup(request: SignupRequest):
    if request.username in USERS:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Store in memory
    USERS[request.username] = request.password
    return {
        "success": True,
        "message": "Account created successfully",
        "token": "dummy-token-injection-molding-assistant"
    }

@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    stored_password = USERS.get(request.username)
    if stored_password and stored_password == request.password:
        return {
            "success": True,
            "message": "Login successful",
            "token": "dummy-token-injection-molding-assistant"
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

