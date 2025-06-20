from typing import List, Optional
from pydantic import BaseModel, Field


class SaveUserCollectionRequest(BaseModel):
    user_id: str
    session_id: str
    content: str


class QueryUserCollectionRequest(BaseModel):
    queries: List[str]
    user_id: str
    session_id: str


class WebSearchQueryData(BaseModel):
    title: str
    url: str
    content: str
    score: float
    raw_content: Optional[str] = Field(default=None, example=None)


class SaveWebCollectionRequest(BaseModel):
    query: str
    follow_up_questions: Optional[str] = None
    answer: Optional[str] = None
    images: List[str]
    results: List[WebSearchQueryData]
    response_time: float

    class Config:
        schema_extra = {
            "example": {
                "query": "What are the current costs associated with treatments for Lupus Nephritis?",
                "follow_up_questions": None,
                "answer": None,
                "images": [
                    "https://www.uspharmacist.com/CMSImagesContent/2012/6/USP1206-Lupus-T2.gif"
                ],
                "results": [
                    {
                        "title": "Treatment Patterns and Health Care Costs of Lupus Nephritis in a United ...",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7319534/",
                        "content": "Corticosteroid use and adverse events known to be associated with corticosteroids were common in this cohort.",
                        "score": 0.37614232,
                        "raw_content": None,
                    }
                ],
                "response_time": 5.53,
            }
        }


class QueryWebCollectionRequest(BaseModel):
    queries: List[str]
    therapeutic_area: str
    domains: List[str]
    include_images: bool
