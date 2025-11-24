from fastapi import FastAPI

# Create the app instance - this is crucial
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Root endpoint working"}

@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "endpoint": "health"}

@app.get("/api/v1/jobs")
def get_jobs():
    return {"jobs": ["Software Engineer", "Data Scientist"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
