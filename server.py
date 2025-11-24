from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"message": "HELLO! Server is WORKING! 🎉"}

@app.get("/test")
def test():
    return {"status": "success", "api": "working"}

# THIS IS THE KEY - Run it directly
if __name__ == "__main__":
    print("⭐ SERVER STARTING...")
    print("⭐ Open browser to: http://localhost:8000")
    print("⭐ Or run: curl http://localhost:8000/")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
