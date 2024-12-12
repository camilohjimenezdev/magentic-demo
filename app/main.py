from fastapi import FastAPI
from .database.database import init_db
from .api.endpoints import router


def create_app():
    app = FastAPI()
    app.include_router(router)
    init_db()
    return app


# Define the app instance at the module level
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
