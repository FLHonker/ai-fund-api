import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from src.agents.fundamentals import fundamentals_agent
from src.agents.market_data import market_data_agent
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.risk_manager import risk_management_agent
from src.agents.sentiment import sentiment_agent
from src.agents.state import AgentState
from src.agents.valuation import valuation_agent
from datetime import datetime, timedelta

# Initialize FastAPI app
app = FastAPI()

# Define workflow
workflow = StateGraph(AgentState)
workflow.add_node("market_data_agent", market_data_agent)
workflow.add_node("technical_analyst_agent", technical_analyst_agent)
workflow.add_node("fundamentals_agent", fundamentals_agent)
workflow.add_node("sentiment_agent", sentiment_agent)
workflow.add_node("risk_management_agent", risk_management_agent)
workflow.add_node("portfolio_management_agent", portfolio_management_agent)
workflow.add_node("valuation_agent", valuation_agent)
workflow.set_entry_point("market_data_agent")
workflow.add_edge("market_data_agent", "technical_analyst_agent")
workflow.add_edge("market_data_agent", "fundamentals_agent")
workflow.add_edge("market_data_agent", "sentiment_agent")
workflow.add_edge("market_data_agent", "valuation_agent")
workflow.add_edge("technical_analyst_agent", "risk_management_agent")
workflow.add_edge("fundamentals_agent", "risk_management_agent")
workflow.add_edge("sentiment_agent", "risk_management_agent")
workflow.add_edge("valuation_agent", "risk_management_agent")
workflow.add_edge("risk_management_agent", "portfolio_management_agent")
workflow.add_edge("portfolio_management_agent", END)

compiled_app = workflow.compile()


class HedgeFundRequest(BaseModel):
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    show_reasoning: Optional[bool] = False
    num_of_news: Optional[int] = Field(
        default=5, ge=1, le=100, description="Number of news articles to analyze (1-100)")
    initial_capital: Optional[float] = Field(
        default=100000.0, ge=0, description="Initial cash amount")
    initial_position: Optional[int] = Field(
        default=0, ge=0, description="Initial stock position")


@app.post("/pred")
async def pred(request: HedgeFundRequest):
    try:
        start_date = request.start_date or (
            datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')
        end_date = request.end_date or (
            datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        if datetime.strptime(start_date, "%Y-%m-%d") > datetime.strptime(end_date, "%Y-%m-%d"):
            raise HTTPException(
                status_code=400, detail="Start date cannot be after end date.")

        portfolio = {
            "cash": request.initial_capital,
            "stock": request.initial_position
        }
        
        final_state = compiled_app.invoke({
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.")
            ],
            "data": {
                "ticker": request.ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "num_of_news": request.num_of_news,
            },
            "metadata": {
                "show_reasoning": request.show_reasoning,
            }
        })
        response_content = final_state["messages"][-1].content
        try:
            json_response = json.loads(response_content[8:-4])
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500, detail="Invalid response format received from agents.")

        return json_response

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Value error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def health_check():
    return {"status": "Service is up and running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
