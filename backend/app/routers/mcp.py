from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from ..database import get_db
from ..models import QueryLog, SystemConfig
from ..services.qdrant_service import qdrant_service
from ..services.yield_service import yield_service

router = APIRouter(prefix="/api/mcp", tags=["Model Context Protocol (MCP)"])

# Standard MCP tools definition
MCP_TOOLS = [
    {
        "name": "search_agricultural_knowledge",
        "description": "Search the FarmerVision agricultural RAG document collections for crop advice, government schemes, and disease remedies.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query in Hindi, English, or Hinglish"},
                "crop": {"type": "string", "description": "Filter by crop name (e.g. wheat, rice, maize, mango)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "predict_crop_yield",
        "description": "Predict estimated crop yield and gross revenue based on historical statistics in Uttar Pradesh districts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "crop": {"type": "string", "description": "Crop name (e.g. wheat, rice)"},
                "district": {"type": "string", "description": "Uttar Pradesh district name"},
                "area_ha": {"type": "number", "description": "Farm area in hectares"}
            },
            "required": ["crop", "district", "area_ha"]
        }
    },
    {
        "name": "get_system_metrics",
        "description": "Fetch overall system telemetry metrics including total query volume, average latencies, satisfaction scores, and guardrail rates.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]

@router.get("/tools")
def list_mcp_tools():
    """List available MCP tools for the AI agent."""
    return {"tools": MCP_TOOLS}

@router.post("/tools/call")
def call_mcp_tool(
    name: str = Body(..., embed=True),
    arguments: Dict[str, Any] = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """Execute an MCP tool call by name and return standard MCP contents."""
    if name == "search_agricultural_knowledge":
        query = arguments.get("query")
        crop = arguments.get("crop")
        
        if not query:
            raise HTTPException(status_code=400, detail="Missing query argument.")
            
        # Run retrieval
        intents = ["disease_pest"] if crop else ["general"]
        hits, tier, score = qdrant_service.retrieve(query, intents=intents)
        
        # Format results
        formatted_hits = [
            f"[{i+1}] (Score: {h['score']:.3f}, Crop: {h['crop']})\n{h['text']}"
            for i, h in enumerate(hits[:3])
        ]
        
        content_text = f"Search Results for '{query}' (Relevance Tier: {tier}, Top Score: {score:.3f}):\n\n" + "\n\n".join(formatted_hits)
        return {
            "content": [{"type": "text", "text": content_text}],
            "isError": False
        }
        
    elif name == "predict_crop_yield":
        crop = arguments.get("crop")
        district = arguments.get("district")
        area_ha = arguments.get("area_ha")
        
        if not all([crop, district, area_ha]):
            raise HTTPException(status_code=400, detail="Missing crop, district, or area_ha arguments.")
            
        pred_t_ha, total_yield = yield_service.predict(crop, district, area_ha)
        economics = yield_service.estimate_profitability(crop, total_yield, area_ha)
        
        content_text = (
            f"Yield Projection for {crop.upper()} in {district.title()} ({area_ha} hectares):\n"
            f"- Yield per Hectare: {pred_t_ha:.2f} t/ha\n"
            f"- Total Projected Yield: {total_yield:.2f} tonnes\n"
            f"- Projected Cultivation Cost: INR {economics['total_cost']:,}\n"
            f"- Projected Gross Revenue (MSP): INR {economics['total_revenue']:,}\n"
            f"- Net Profit Margin: INR {economics['net_profit']:,} (ROI: {economics['roi_percent']}%)\n"
        )
        return {
            "content": [{"type": "text", "text": content_text}],
            "isError": False
        }
        
    elif name == "get_system_metrics":
        from .admin import get_stats
        stats = get_stats(db)
        summary = stats.get("summary", {})
        
        content_text = (
            f"FarmerVision System Telemetry:\n"
            f"- Total Client Queries: {summary.get('total_queries')}\n"
            f"- Average Request Latency: {summary.get('average_latency_ms')} ms\n"
            f"- Customer Satisfaction: {summary.get('satisfaction_rate')}%\n"
            f"- Safety/Guardrail Interceptions: {summary.get('blocked_queries')} ({summary.get('safety_violation_rate')}%)\n"
        )
        return {
            "content": [{"type": "text", "text": content_text}],
            "isError": False
        }
        
    else:
        raise HTTPException(status_code=404, detail=f"MCP Tool '{name}' not found.")

@router.post("/rpc")
def mcp_json_rpc(request: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    """
    Standard JSON-RPC 2.0 entrypoint for MCP clients (like Claude Desktop or Cursor).
    Supports methods: 'tools/list', 'tools/call'.
    """
    method = request.get("method")
    params = request.get("params", {})
    req_id = request.get("id")
    
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "result": {"tools": MCP_TOOLS},
            "id": req_id
        }
    elif method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            res = call_mcp_tool(name, arguments, db)
            return {
                "jsonrpc": "2.0",
                "result": res,
                "id": req_id
            }
        except HTTPException as he:
            return {
                "jsonrpc": "2.0",
                "error": {"code": he.status_code, "message": he.detail},
                "id": req_id
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)},
                "id": req_id
            }
    else:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": f"Method {method} not found"},
            "id": req_id
        }
