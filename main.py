import os
import json
import asyncio
import sys
import subprocess
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled,InputGuardrailTripwireTriggered

# Load environment variables
load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError("Please set BASE_URL, API_KEY, and MODEL_NAME.")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)

# --- Structured Models ---
class TrendingNews(BaseModel):
    category: str
    headlines: List[str]

class ClaimVerification(BaseModel):
    claim: str
    verdict: str
    evidence: List[str]

class NewsSummary(BaseModel):
    topic: str
    summary_bullets: List[str]

@dataclass
class UserContext:
    user_id: str
    preferred_categories: List[str] = None
    session_start: datetime = None

    def __post_init__(self):
        if self.preferred_categories is None:
            self.preferred_categories = []
        if self.session_start is None:
            self.session_start = datetime.now()


# --- Tools ---
@function_tool
def get_trending_news(topic: str) -> str:
    """Simulate fetching trending headlines for a given topic."""
    trending = {
        "AI": [
            "Meta releases open-source LLM",
            "Elon Musk updates Grok AI",
            "OpenAI debuts GPT-5 teaser"
        ],
        "Politics": [
            "Elections in Germany spark debate",
            "US Senate passes budget bill"
        ],
        "Finance": [
            "Bitcoin hits new high",
            "Apple stock soars after earnings"
        ]
    }
    return json.dumps({"category": topic, "headlines": trending.get(topic, ["No trending news found."])} )

@function_tool
def fact_check_claim(claim: str) -> str:
    """Simulate RAG-based fact checking for a claim."""
    dummy_results = {
        "Did Apple acquire OpenAI?": {
            "verdict": "Unverified",
            "evidence": [
                "No official press release found.",
                "Rumors suggest talks but no confirmation.",
                "Analyst reports mention strategic discussions."
            ]
        }
    }
    result = dummy_results.get(claim, {
        "verdict": "Insufficient evidence",
        "evidence": ["No substantial sources found."]
    })
    result["claim"] = claim
    return json.dumps(result)

@function_tool
def summarize_news(article_text: str) -> str:
    """Simulate news summarization into 3–5 bullet points."""
    summary = [
        "Headline recap of the article.",
        "Key background information.",
        "Major developments explained.",
        "Impacts or implications outlined.",
        "Expert opinion or conclusion."
    ]
    return json.dumps({"topic": "Custom Article", "summary_bullets": summary})

# --- Agents ---
trending_agent = Agent[UserContext](
    name="Trending News Agent",
    handoff_description="Fetches and ranks trending news topics",
    instructions="""
    You specialize in detecting trending news topics by category.
    Use get_trending_news tool and return grouped headlines.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[get_trending_news],
    output_type=TrendingNews
)

fact_checker_agent = Agent[UserContext](
    name="Fact Checker Agent",
    handoff_description="Verifies factual accuracy of user claims",
    instructions="""
    Use fact_check_claim tool to check veracity of claims using RAG simulation.
    Provide a clear verdict and list supporting evidence.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[fact_check_claim],
    output_type=ClaimVerification
)

summarizer_agent = Agent[UserContext](
    name="News Summarizer Agent",
    handoff_description="Summarizes long news articles into concise bullets",
    instructions="""
    Use summarize_news to create readable 3–5 bullet point summaries.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[summarize_news],
    output_type=NewsSummary
)

conversation_agent = Agent[UserContext](
    name="NewsSense Controller",
    handoff_description="Entry point agent for detecting intent and routing",
    instructions="""
    You are the main conversation agent that determines user intent:
    - 'Get Trending' → route to Trending News Agent
    - 'Verify Claim' → route to Fact Checker Agent
    - 'Summarize Topic' → route to News Summarizer Agent

    Be concise, route appropriately, and explain which specialist is being used.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    handoffs=[trending_agent, fact_checker_agent, summarizer_agent]
)

# --- Main Execution ---
async def main():
    user_context = UserContext(user_id="news_user_1", preferred_categories=["AI", "Politics"])
    queries = [
        "What's trending in AI today?",
        "Did Apple acquire OpenAI?",
        "Summarize this article about the latest AI model."
    ]

    for query in queries:
        print("\n" + "="*50)
        print(f"QUERY: {query}")

        try:
            result = await Runner.run(conversation_agent, query, context=user_context)

            print("\nFINAL RESPONSE:")

            if isinstance(result.final_output, TrendingNews):
                print(f"\n📢 TRENDING IN {result.final_output.category.upper()} 📢")
                for headline in result.final_output.headlines:
                    print(f"- {headline}")

            elif isinstance(result.final_output, ClaimVerification):
                print(f"\n🔍 CLAIM VERIFICATION: {result.final_output.claim}")
                print(f"Verdict: {result.final_output.verdict}")
                print("Evidence:")
                for evidence in result.final_output.evidence:
                    print(f"- {evidence}")

            elif isinstance(result.final_output, NewsSummary):
                print(f"\n📰 SUMMARY OF: {result.final_output.topic}")
                for bullet in result.final_output.summary_bullets:
                    print(f"- {bullet}")

            else:
                print(result.final_output)
        except InputGuardrailTripwireTriggered:
             print(" → ⚠️ Guardrail triggered. Input was blocked.")
if __name__ == "__main__":
    asyncio.run(main())