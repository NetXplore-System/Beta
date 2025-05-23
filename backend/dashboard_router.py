import os
import json
import logging
import uuid
from typing import List, Optional
from pydantic import BaseModel

import networkx as nx
from community import community_louvain
from networkx.algorithms import community as nx_community

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from database import get_db
from models import Research, ResearchFilter, NetworkAnalysis, Comparisons
from auth_router import get_current_user
from utils import apply_comparison_filters, find_common_nodes, mark_common_nodes, get_network_metrics

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/dashboard/{user_id}")
async def get_dashboard_data(
    user_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        user_uuid = uuid.UUID(user_id)
        
        query = select(
            Research.research_id,
            Research.research_name,
            Research.description,
            Research.platform,
            Research.created_at,
            NetworkAnalysis.nodes,
            NetworkAnalysis.links
        ).select_from(
            Research.__table__.join(
                NetworkAnalysis.__table__, 
                Research.research_id == NetworkAnalysis.research_id,
                isouter=True
            )
        ).where(Research.user_id == user_uuid).order_by(Research.created_at.desc())
        
        result = await db.execute(query)
        rows = result.fetchall()
        
        researches = []
        for row in rows:
            nodes_count = len(row.nodes) if row.nodes else 0
            communities_count = 0
            
            if row.nodes:
                communities = set()
                for node in row.nodes:
                    if 'community' in node:
                        communities.add(node['community'])
                communities_count = len(communities)
            
            research_data = {
                "id": str(row.research_id),
                "name": row.research_name,
                "description": row.description or "No description",
                "type": row.platform,
                "date": row.created_at.strftime("%Y-%m-%d") if row.created_at else "",
                "nodes": nodes_count,
                "communities": communities_count
            }
            researches.append(research_data)
        
        total_researches = len(researches)
        whatsapp_count = sum(1 for r in researches if r["type"] == "whatsapp")
        wikipedia_count = sum(1 for r in researches if r["type"] == "wikipedia")
        total_nodes = sum(r["nodes"] for r in researches)
        
        return JSONResponse(content={
            "researches": researches,
            "stats": {
                "total_researches": total_researches,
                "whatsapp_researches": whatsapp_count,
                "wikipedia_researches": wikipedia_count,
                "total_nodes": total_nodes
            }
        }, status_code=200)
        
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))