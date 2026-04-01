import uvicorn

if __name__ == "__main__":
    uvicorn.run("gigachat_openai_proxy.main:app", host="0.0.0.0", port=8000)
