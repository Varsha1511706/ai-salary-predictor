from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Working on port 9000!"}

if __name__ == "__main__":
    print("🚀 Starting on port 9000...")
    uvicorn.run(app, host="127.0.0.1", port=9000)
