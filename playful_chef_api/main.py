from fastapi import FastAPI

# Create FastAPI application instance
app = FastAPI(
    title="Sample FastAPI App",
    description="A simple FastAPI application with one handler",
    version="1.0.0",
)


# Define a simple handler
@app.get("/")
async def read_root():
    return {"message": "Welcome to FastAPI!", "status": "success"}


# Add another handler for demonstration
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello, {name}!", "greeting": "Welcome to our API"}
