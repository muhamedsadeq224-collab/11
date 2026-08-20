import os
import json
import re
from typing import TypedDict, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from langgraph.graph import StateGraph, END

app = FastAPI(title="Roblox LangGraph AI Agent")

# الاتصال بـ Gemini API عبر المفتاح الموجود في بيئة Render
ai = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"

class AgentState(TypedDict):
    user_prompt: str
    plan: str
    build_data: List[Dict[str, Any]]
    luau_code: str
    is_code_valid: bool
    error_message: str
    retry_count: int

# 1. القائد (Planner)
def lead_planner_node(state: AgentState) -> Dict:
    prompt = f"أنت قائد فريق Roblox Studio. حلل الطلب في خطة مختصرة جداً: '{state['user_prompt']}'"
    response = ai.models.generate_content(model=MODEL_NAME, contents=prompt)
    return {"plan": response.text}

# 2. المصمم (Builder Parts)
def builder_node(state: AgentState) -> Dict:
    prompt = f"""أنت مصمم مجسمات Roblox Studio.
    بناءً على الطلب: '{state['user_prompt']}'
    ولد مصفوفة JSON فقط بالشكل التالي بدون أي شرح أو ماركداون:
    [
      {{"name": "BasePart", "size": [10, 1, 10], "position": [0, 5, 0], "color": [255, 0, 0]}}
    ]"""
    response = ai.models.generate_content(model=MODEL_NAME, contents=prompt)
    clean_text = re.sub(r'
    try:
        data = json.loads(clean_text)
    except Exception:
        data = []
    return {"build_data": data}

# 3. المبرمج (Luau Coder)
def coder_node(state: AgentState) -> Dict:
    feedback = f"\nاصلاح الخطأ السابق: {state.get('error_message')}" if state.get('error_message') else ""
    prompt = f"""أنت خبير Luau في Roblox Studio.
    اكتب كود Luau ينفذ الطلب: '{state['user_prompt']}'.{feedback}
    اكتب كود صافي فقط بدون 
lua وبدون أي كلام جانبي."""
    response = ai.models.generate_content(model=MODEL_NAME, contents=prompt)
    clean_code = re.sub(r'`lua|```', '', response.text).strip()
    return {"luau_code": clean_code, "retry_count": state.get("retry_count", 0) + 1}

# 4. الفاحص (Validator)
def validator_node(state: AgentState) -> Dict:
    prompt = f"افحص كود Luau التالي لـ Roblox، إذا كان صحيحاً أجب بـ 'VALID' فقط، وإذا فيه خطأ اشرحه باختصار:\n{state['luau_code']}"
    response = ai.models.generate_content(model=MODEL_NAME, contents=prompt)
    if "VALID" in response.text.upper():
        return {"is_code_valid": True, "error_message": ""}
    return {"is_code_valid": False, "error_message": response.text}

def should_continue(state: AgentState) -> str:
    if state["is_code_valid"] or state.get("retry_count", 0) >= 3:
        return "end"
    return "retry"

# بناء المخطط (Workflow)
workflow = StateGraph(AgentState)
workflow.add_node("lead_planner", lead_planner_node)
workflow.add_node("builder", builder_node)
workflow.add_node("coder", coder_node)
workflow.add_node("validator", validator_node)

workflow.set_entry_point("lead_planner")
workflow.add_edge("lead_planner", "builder")
workflow.add_edge("builder", "coder")
workflow.add_edge("coder", "validator")
workflow.add_conditional_edges("validator", should_continue, {"end": END, "retry": "coder"})

app_graph = workflow.compile()

class PromptInput(BaseModel):
    prompt: str

@app.post("/generate")
def generate_roblox_assets(data: PromptInput):
    try:
        initial_state = {
            "user_prompt": data.prompt, "plan": "", "build_data": [],
            "luau_code": "", "is_code_valid": False, "error_message": "", "retry_count": 0
        }
        final_state = app_graph.invoke(initial_state)
        return {
            "success": True,
            "plan": final_state["plan"],
            "build": final_state["build_data"],
            "code": final_state["luau_code"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
