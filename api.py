import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

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
    
import json
import uuid
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Pydantic Models ----------

class BrandGenerateRequest(BaseModel):
    brandName: str
    brandDomain: str
    category: str
    brandDescription: str
    market: str
    userTags: List[str] = []

class MonitoringPrompt(BaseModel):
    prompt_id: Optional[str] = None
    text: str
    level: str
    tags: List[str] = []
    market: str
    language: str = "en"
    is_active: bool = True

class Competitor(BaseModel):
    competitor_id: Optional[str] = None
    label: str
    type: str = "generic"

class BrandConfigSaveRequest(BaseModel):
    target_brand: str
    brand_domain: str
    category: str
    brand_description: str
    market: str
    language: str = "en"
    user_tags: List[str] = []
    generated_claims: List[str] = []
    generated_positioning: str = ""
    competitors: List[Competitor] = []
    monitoring_prompts: List[MonitoringPrompt] = []


# ---------- Helper: LLM 生成 ----------

def generate_with_openai(system_prompt: str, user_prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


def generate_competitors(brand_name: str, category: str, market: str) -> list:
    system_prompt = "You are a market research assistant. Return only valid JSON, no markdown, no explanation."
    user_prompt = f"""
Research and generate 5 real competitor brands for "{brand_name}" in the "{category}" category for the "{market}" market.

Rules:
- Return REAL brand names that actually exist and sell "{category}" products
- These should be genuine competitors that a consumer would compare against "{brand_name}"
- Do not use generic descriptions like "Budget-Friendly Option" or "Premium Brand"
- Include a mix of: market leaders, niche players, and direct competitors

Return a JSON array of exactly 5 items:
[{{"label": "ActualBrandName", "type": "brand"}}]
"""
    raw = generate_with_openai(system_prompt, user_prompt)
    competitors = json.loads(raw)
    for i, c in enumerate(competitors):
        c["competitor_id"] = f"comp_{str(i+1).zfill(3)}"
    return competitors

def generate_prompts(brand_name: str, category: str, market: str, user_tags: list, brand_domain: str) -> list:
    brand_slug = brand_domain.replace(".com", "").replace(".", "_")
    
    system_prompt = "You are an AI monitoring prompt generator. Return only valid JSON, no markdown, no explanation."
    user_prompt = f"""
Generate exactly 20 search monitoring prompts for brand "{brand_name}" in category "{category}", market "{market}".
Tags to consider: {user_tags}

Generate exactly 5 prompts for each of these 4 levels:

L1 - Unbranded problem/symptom. No category named. User describes a problem that the product solves, but never mentions the category.
Example: "I can't sleep because of my partner's snoring, what can I do?"

L2 - Category only. Broad commercial intent. Category is named but no brand or specific attributes.
Example: "best earplugs for sleeping"

L3 - Category + attribute. Narrow enough that a brand with the right attributes should surface. Category + specific attribute/use case.
Example: "best reusable silicone earplugs for side sleepers"

L4 - Branded. Brand name "{brand_name}" is explicitly mentioned. Tests what the AI says about the brand.
Example: "what do you think of {brand_name} earplugs?"

Return a JSON array of exactly 20 items, each item:
{{"text": "...", "level": "L1"|"L2"|"L3"|"L4"}}

Make sure there are exactly 5 items per level.
"""
    raw = generate_with_openai(system_prompt, user_prompt)
    prompts = json.loads(raw)
    
    level_counters = {"L1": 1, "L2": 1, "L3": 1, "L4": 1}
    level_labels = {
        "L1": "Unbranded problem",
        "L2": "Category",
        "L3": "Category + attribute",
        "L4": "Branded"
    }
    result = []
    for p in prompts:
        level = p.get("level", "L2")
        count = level_counters.get(level, 1)
        result.append({
            "prompt_id": f"{level}_{brand_slug}_{market}_{str(count).zfill(3)}",
            "text": p["text"],
            "level": level,
            "level_label": level_labels.get(level, "Category"),
            "tags": [],
            "market": market,
            "language": "en",
            "is_active": True
        })
        level_counters[level] = count + 1
    return result

def generate_claims_and_positioning(brand_name: str, brand_description: str, category: str) -> dict:
    system_prompt = "You are a brand strategist. Return only valid JSON, no markdown, no explanation."
    user_prompt = f"""
Based on this brand:
- Name: {brand_name}
- Category: {category}
- Description: {brand_description}

Generate:
1. 4 short factual brand claims
2. 1 positioning statement (2-3 sentences)

Return JSON only: {{"claims": ["...", "..."], "positioning": "..."}}
"""
    raw = generate_with_openai(system_prompt, user_prompt)
    return json.loads(raw)


# ---------- Routes ----------

@app.post("/api/v1/brand/generate")
def brand_generate(request: BrandGenerateRequest):
    try:
        competitors = generate_competitors(
            request.brandName, request.category, request.market
        )
        prompts = generate_prompts(
    request.brandName, request.category, request.market, request.userTags, request.brandDomain
	)
        claims_and_positioning = generate_claims_and_positioning(
            request.brandName, request.brandDescription, request.category
        )
        return {
            "success": True,
            "data": {
                "competitors": competitors,
                "monitoring_prompts": prompts,
                "generated_claims": claims_and_positioning.get("claims", []),
                "generated_positioning": claims_and_positioning.get("positioning", "")
            }
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"LLM returned invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/brand/config/save")
def brand_config_save(request: BrandConfigSaveRequest):
    try:
        now = datetime.now(timezone.utc)

        # 为 prompt_id 为 None 的条目分配新 id
        prompts = []
        level_new_counters = {}
        for p in request.monitoring_prompts:
            p_dict = p.dict()
            if not p_dict["prompt_id"]:
                level = p_dict["level"]
                level_new_counters[level] = level_new_counters.get(level, 0) + 1
                p_dict["prompt_id"] = f"{level}_{request.market}_new_{str(level_new_counters[level]).zfill(3)}"
            prompts.append(p_dict)

        # 为 competitor_id 为 None 的条目分配新 id
        competitors = []
        for i, c in enumerate(request.competitors):
            c_dict = c.dict()
            if not c_dict["competitor_id"]:
                c_dict["competitor_id"] = f"comp_new_{str(i+1).zfill(3)}"
            competitors.append(c_dict)

        doc = {
            "target_brand": request.target_brand,
            "brand_domain": request.brand_domain,
            "category": request.category,
            "brand_description": request.brand_description,
            "market": request.market,
            "language": request.language,
            "user_tags": request.user_tags,
            "generated_claims": request.generated_claims,
            "generated_positioning": request.generated_positioning,
            "competitors": competitors,
            "monitoring_prompts": prompts,
            "updated_at": now,
        }

        result = db["brand_configs"].update_one(
            {"brand_domain": request.brand_domain},
            {
                "$set": doc,
                "$setOnInsert": {"created_at": now}
            },
            upsert=True
        )

        return {
            "success": True,
            "message": "Brand config saved successfully.",
            "upserted": result.upserted_id is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/brand/config/{brand_domain}")
def brand_config_get(brand_domain: str):
    config = db["brand_configs"].find_one(
        {"brand_domain": brand_domain},
        {"_id": 0}
    )
    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"No config found for brand_domain={brand_domain}"
        )
    return {"success": True, "data": config}
