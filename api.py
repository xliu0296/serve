import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

app = FastAPI(title="AI Brand Scoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo阶段用 *；正式部署可改成前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "ai_brand_scoring"

if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable is not set")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]


@app.get("/")
def home():
    return {"message": "AI Brand Scoring API is running"}


@app.get("/health")
def health_check():
    try:
        client.admin.command("ping")
        return {"status": "ok", "mongodb": "connected"}
    except Exception as e:
        return {"status": "error", "mongodb": str(e)}


@app.get("/aggregation")
def get_aggregation(
    week_id: str = "2026-W17",
    brand: str = "HappyEars"
):
    result = db["aggregation_summaries"].find_one(
        {"week_id": week_id, "brand": brand},
        {"_id": 0}
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No aggregation found for week_id={week_id}, brand={brand}"
        )

    return result


@app.get("/composite")
def get_composite_scores(
    week_id: str = "2026-W17",
    brand: str = "HappyEars"
):
    results = list(
        db["composite_scores"].find(
            {"week_id": week_id, "brand": brand},
            {"_id": 0}
        )
    )

    return {
        "week_id": week_id,
        "brand": brand,
        "count": len(results),
        "data": results
    }