import os
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Form, Query, Depends, HTTPException
from fastapi.responses import JSONResponse

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete

from pydantic import BaseModel

from database import get_db
from models import Research, Message, ResearchFilter, NetworkAnalysis, Comparisons
from auth_router import get_current_user
from wikipedia_router import analyze_wikipedia
from analyzers.factory import get_analyzer
from utils import calculate_comparison_stats


UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads/")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logger = logging.getLogger(__name__)

router = APIRouter()

class NetworkAnalysisData(BaseModel):
    nodes: List[dict]
    links: List[dict]
    metric_name: Optional[str] = None

@router.post("/save-research")
async def save_research(
    file_name: str = Form(...),
    researcher_id: str = Form(...),
    research_name: str = Form(...),
    description: Optional[str] = Form(None),
    comparison_data: Optional[str] = Form(None),
    comparison_filters: Optional[str] = Form(None),
    platform: str = Form(...),
    selected_metric: str = Form(None),
    start_date: str = Query(None),
    end_date: str = Query(None), 
    start_time: str = Query(None),
    end_time: str = Query(None),
    limit: int = Query(None),
    limit_type: str = Query("first"), 
    min_length: int = Query(None),
    max_length: int = Query(None),
    keywords: str = Query(None),
    min_messages: int = Query(None),
    max_messages: int = Query(None),
    active_users: int = Query(None),
    selected_users: str = Query(None),
    username: str = Query(None),
    anonymize: bool = Query(False),
    include_messages: bool = Query(True),
    directed: bool = Query(False),
    use_history: bool = Query(False),
    normalize: bool = Query(False),
    history_length: int = Query(3),
    message_weights: str = Query([5,3,2]),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    communities: Optional[str] = Form(None),
):
    try:
        file_path = None
        if platform == "whatsapp":
            file_path = os.path.join(UPLOAD_FOLDER, file_name)
            if not os.path.exists(file_path):
                logger.error(f"File '{file_name}' not found.")
                raise HTTPException(status_code=404, detail=f"File '{file_name}' not found.")
        
        parsed_message_weights = None
        if message_weights:
            try:
                parsed_message_weights = json.loads(message_weights)
                if not isinstance(parsed_message_weights, list) or not all(isinstance(x, (int, float)) for x in parsed_message_weights):
                    raise ValueError("message_weights must be a list of numbers")
            except Exception as e:
                logger.warning(f"Invalid message_weights format: {message_weights}, error: {e}")
                parsed_message_weights = [0.5, 0.3, 0.2] if history_length == 3 else [0.7, 0.3]
                
        data = None
        if platform == "wikipedia":
            data = await analyze_wikipedia(
                section_title=file_name,
                limit=int(limit) if limit is not None else None,
                limit_type=limit_type,
                min_length=int(min_length) if min_length is not None else None,
                max_length=int(max_length) if max_length is not None else None,
                keywords=keywords,
                min_messages=int(min_messages) if min_messages is not None else None,
                max_messages=int(max_messages) if max_messages is not None else None,
                active_users=int(active_users) if active_users is not None else None,
                selected_users=selected_users,
                username=username,
                anonymize=False,
                directed=directed,
                use_history=use_history,
                normalize=normalize,
                start_date=start_date,
                end_date=end_date,
                start_time=start_time,
                end_time=end_time,
                history_length=int(history_length) if history_length is not None else 3,
                message_weights=parsed_message_weights,
                include_messages=include_messages,
            )
        else:
            analyzer = get_analyzer(platform)
            data = await analyzer.analyze(
                filename=file_name,
                limit=int(limit) if limit is not None else None,
                limit_type=limit_type,
                min_length=int(min_length) if min_length is not None else None,
                max_length=int(max_length) if max_length is not None else None,
                keywords=keywords,
                min_messages=int(min_messages) if min_messages is not None else None,
                max_messages=int(max_messages) if max_messages is not None else None,
                active_users=int(active_users) if active_users is not None else None,
                selected_users=selected_users,
                username=username,
                anonymize=False,
                directed=directed,
                use_history=use_history,
                normalize=normalize,
                start_date=start_date,
                end_date=end_date,
                start_time=start_time,
                end_time=end_time,
                history_length=int(history_length) if history_length is not None else 3,
                message_weights=parsed_message_weights,
                is_for_save=True
            )
        
        if isinstance(data, dict):
            pass  
        elif hasattr(data, "body"):
            data = json.loads(data.body)
        elif isinstance(data, str):
            data = json.loads(data)
        else:
            logger.error("Unsupported response type from analyzer.analyze()")
            raise HTTPException(status_code=500, detail="Invalid response from analyzer")


        if not data or "nodes" not in data or "links" not in data:
            logger.error("Invalid data format received from analysis.")
            raise HTTPException(status_code=400, detail="Invalid data format received from analysis.")
        
        
        new_research = Research(
            research_name=research_name,
            description=description,
            user_id=researcher_id,
            platform=platform,
        )
        db.add(new_research)
        await db.commit()
        await db.refresh(new_research)

        if include_messages and isinstance(data.get("messages"), list):
            for message in data["messages"]:
                new_message = Message(
                    research_id=new_research.research_id,
                    message_text=message[1],
                    send_by=message[0],
                )
                db.add(new_message)
            await db.commit()

        new_filter = ResearchFilter(
            research_id=new_research.research_id,
            start_date=start_date,
            end_date=end_date,
            start_time=start_time,  
            end_time=end_time,
            message_limit=limit,
            limit_type=limit_type,
            min_message_length=min_length,
            max_message_length=max_length,
            keywords=keywords,
            min_messages=min_messages,
            max_messages=max_messages,
            top_active_users=active_users,
            selected_users=selected_users,
            filter_by_username=username,
            anonymize=anonymize,
            directed=directed,
            use_history=use_history,
            normalize=normalize,
            history_length=history_length if use_history else None,
            message_weights=parsed_message_weights if use_history else None,
        )
        db.add(new_filter) 
        parsed_communities = json.loads(communities) if communities not in [None, "", "[]"] else []
        new_analysis = NetworkAnalysis(
            research_id=new_research.research_id,
            nodes=data['nodes'],
            links=data['links'],
            is_connected=data['is_connected'],
            metric_name=selected_metric,
            communities=parsed_communities

        )
        db.add(new_analysis)
        await db.commit()
        await db.refresh(new_analysis)

        if comparison_data:
            try:
                comparison_data = json.loads(comparison_data)
                comparison_filters = json.loads(comparison_filters)
                
                for comp_data, comp_filter in zip(comparison_data, comparison_filters):
                    
                    comp_message_weights = comp_filter.get("config", {}).get("messageWeights")
                    parsed_comp_message_weights = None
                    if comp_message_weights:
                        try:
                            if isinstance(comp_message_weights, list):
                                parsed_comp_message_weights = comp_message_weights
                            else:
                                parsed_comp_message_weights = json.loads(comp_message_weights)
                            if not isinstance(parsed_comp_message_weights, list) or not all(isinstance(x, (int, float)) for x in parsed_comp_message_weights):
                                raise ValueError("message_weights must be a list of numbers")
                        except Exception as e:
                            logger.warning(f"Invalid comparison message_weights format: {comp_message_weights}, error: {e}")
                            comp_history_length = int(comp_filter.get("config", {}).get("messageCount", 3))
                            parsed_comp_message_weights = [0.5, 0.3, 0.2] if comp_history_length == 3 else [0.7, 0.3]

                    messages = None
                    if platform == "wikipedia":
                        print(comp_data.get("file_name", file_name))
                        messages = await analyze_wikipedia(
                            section_title=comp_data.get("file_name", file_name),
                            start_date=comp_filter.get("timeFrame", {}).get("startDate"),
                            end_date=comp_filter.get("timeFrame", {}).get("endDate"),
                            start_time=comp_filter.get("timeFrame", {}).get("startTime"),
                            end_time=comp_filter.get("timeFrame", {}).get("endTime"),
                            limit=comp_filter.get("limit", {}).get("count"),
                            limit_type="last" if comp_filter.get("limit", {}).get("fromEnd") else "first",
                            min_length=comp_filter.get("messageCriteria", {}).get("minLength"),
                            max_length=comp_filter.get("messageCriteria", {}).get("maxLength"),
                            keywords=comp_filter.get("messageCriteria", {}).get("keywords"),
                            min_messages=comp_filter.get("userFilters", {}).get("minMessages"),
                            max_messages=comp_filter.get("userFilters", {}).get("maxMessages"),
                            username=comp_filter.get("userFilters", {}).get("usernameFilter"),
                            selected_users=comp_filter.get("userFilters", {}).get("selectedUsers", ""),
                            active_users=comp_filter.get("userFilters", {}).get("topActiveUsers", 0),
                            anonymize=False,
                            directed=comp_filter.get("config", {}).get("directed", False),
                            use_history=comp_filter.get("config", {}).get("history", False),
                            normalize=comp_filter.get("config", {}).get("normalized", False),
                            history_length=int(comp_filter.get("config", {}).get("messageCount", 3)) if comp_filter.get("config", {}).get("history", False) else None,
                            message_weights=parsed_comp_message_weights if comp_filter.get("config", {}).get("history", False) else None,
                            include_messages=True
                        )
                    else:
                        analyzer = get_analyzer(platform)
                        messages = await analyzer.analyze(
                            filename=comp_data.get("file_name", file_name),
                            start_date=comp_filter.get("timeFrame", {}).get("startDate"),
                            end_date=comp_filter.get("timeFrame", {}).get("endDate"),
                            start_time=comp_filter.get("timeFrame", {}).get("startTime"),
                            end_time=comp_filter.get("timeFrame", {}).get("endTime"),
                            limit=comp_filter.get("limit", {}).get("count"),
                            limit_type="last" if comp_filter.get("limit", {}).get("fromEnd") else "first",
                            min_length=comp_filter.get("messageCriteria", {}).get("minLength"),
                            max_length=comp_filter.get("messageCriteria", {}).get("maxLength"),
                            keywords=comp_filter.get("messageCriteria", {}).get("keywords"),
                            min_messages=comp_filter.get("userFilters", {}).get("minMessages"),
                            max_messages=comp_filter.get("userFilters", {}).get("maxMessages"),
                            username=comp_filter.get("userFilters", {}).get("usernameFilter"),
                            selected_users=comp_filter.get("userFilters", {}).get("selectedUsers", ""),
                            active_users=comp_filter.get("userFilters", {}).get("topActiveUsers", 0),
                            anonymize=False,
                            directed=comp_filter.get("config", {}).get("directed", False),
                            use_history=comp_filter.get("config", {}).get("history", False),
                            normalize=comp_filter.get("config", {}).get("normalized", False),
                            history_length=int(comp_filter.get("config", {}).get("messageCount", 3)) if comp_filter.get("config", {}).get("history", False) else None,
                            message_weights=parsed_comp_message_weights if comp_filter.get("config", {}).get("history", False) else None,
                            is_for_save=True
                        )

                    if isinstance(messages, dict):
                        messages = messages.get("messages", [])
                    elif hasattr(messages, "body"):
                        messages = json.loads(messages.body).get("messages", [])
                    elif isinstance(messages, str):
                        messages = json.loads(messages).get("messages", [])
                    else:
                        logger.warning("Unsupported response type for comparison messages.")
                        messages = []

                    
                    logger.info(f"🔹 Comparison messages extracted successfully: {len(messages)} messages")

                    comparison_stats = calculate_comparison_stats(data["nodes"], comp_data.get("nodes", []), data["links"], comp_data.get("links", []))

                    new_comparison = Comparisons(
                        research_id=new_research.research_id,
                        original_analysis=new_analysis.id,
                        nodes=comp_data.get('nodes', []),
                        links=comp_data.get('links', []),
                        is_connected=comp_data.get('is_connected', True),
                        messages=json.dumps(messages), 
                        filters=json.dumps(comp_filter),
                        statistics=json.dumps(comparison_stats), 
                        file_name=comp_data.get("filename"),
                        original_file_name=file_name 
                    )
                    db.add(new_comparison)
                
                
                await db.commit()
            except json.JSONDecodeError:
                logger.error(f"Invalid comparison data format: {comparison_data}")
                pass

        return {
            "message": "Data saved successfully", 
            "research_id": str(new_research.research_id)
        }

    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving data: {str(e)}")


@router.delete("/research/{research_id}")
async def delete_research(
    research_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        research = await db.get(Research, research_id)
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        if str(research.user_id) != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Not authorized to delete this research")

        await db.execute(
            delete(Comparisons)
            .where(Comparisons.research_id == research_id)
        )

        await db.execute(
            delete(NetworkAnalysis)
            .where(NetworkAnalysis.research_id == research_id)
        )

        await db.execute(
            delete(ResearchFilter)
            .where(ResearchFilter.research_id == research_id)
        )

        await db.execute(
            delete(Message)
            .where(Message.research_id == research_id)
        )

        await db.execute(
            delete(Research)
            .where(Research.research_id == research_id)
        )

        await db.commit()

        return JSONResponse(
            content={"message": f"Research {research_id} and all related data deleted successfully"},
            status_code=200
        )

    except HTTPException as he:
        logger.error(f"Error deleting research: {str(he.detail)}")
        raise he
    except Exception as e:
        logger.error(f"Error deleting research: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting research: {str(e)}"
        )
    


@router.put("/research/{research_id}")
async def update_research_data(
    research_id: str,
    updated_data: dict,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        research = await db.get(Research, research_id)
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        if str(research.user_id) != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Not authorized")

        file_name = updated_data.get("file_name")
        if not file_name:
            if "research_name" in updated_data or "description" in updated_data:
                research.research_name = updated_data.get("research_name", research.research_name)
                research.description = updated_data.get("description", research.description)
                await db.commit()
                await db.refresh(research)
                return JSONResponse(
                    content={
                        "status": "success",
                        "message": "Research name and/or description updated successfully",
                        "data": research.to_dict()
                    },
                    status_code=200
                )
            else:
                raise HTTPException(status_code=400, detail="No updates provided for research name or description.")
        
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        if not os.path.exists(file_path):
            logger.error(f"File '{file_name}' not found.")
            raise HTTPException(status_code=404, detail=f"File '{file_name}' not found.")
    

        logger.info(f"🔹 Lines: {updated_data.get('filters', {})}")

        filters_data = {
            **updated_data.get("filters", {}), 
            "limit": int(updated_data.get("filters", {}).get("message_limit") or 0),
            "min_length": int(updated_data.get("filters", {}).get("min_message_length") or 0),
            "max_length": int(updated_data.get("filters", {}).get("max_message_length") or 0),
            "active_users": int(updated_data.get("filters", {}).get("top_active_users") or 0),
            "min_messages": int(updated_data.get("filters", {}).get("min_messages") or 0),
            "max_messages": int(updated_data.get("filters", {}).get("max_messages") or 0),
            "history_length": int(updated_data.get("filters", {}).get("history_length") or 3),
            "username": updated_data.get("filters", {}).get("filter_by_username"),
            "selected_users": updated_data.get("filters", {}).get("specific_users"),
        }

        filters_data.pop("message_limit", None)
        filters_data.pop("min_message_length", None) 
        filters_data.pop("max_message_length", None)
        filters_data.pop("top_active_users", None)
        filters_data.pop("filter_by_username", None)
        filters_data.pop("specific_users", None)
        analyzer = get_analyzer(research.platform)

        new_data = await analyzer.analyze(
            filename=file_name,
            is_for_save=True,
            **filters_data
        )

        if isinstance(new_data, JSONResponse):
            new_data = new_data.body
            if isinstance(new_data, bytes):
                new_data = json.loads(new_data.decode('utf-8'))
            else:
                new_data = json.loads(new_data)
                
        research.research_name = updated_data.get("research_name", research.research_name)
        research.description = updated_data.get("description", research.description)
        research.file_name = file_name or research.file_name

        filter_query = select(ResearchFilter).where(ResearchFilter.research_id == research.research_id)
        filter_result = await db.execute(filter_query)
        filters = filter_result.scalars().first()

        INT_FIELDS = [
            "message_limit", "min_message_length", "max_message_length",
            "min_messages", "max_messages", "top_active_users", "history_length"
        ]

        if filters:
            for key, value in updated_data.get("filters", {}).items():
                if key in INT_FIELDS and value != "" and value is not None:
                    setattr(filters, key, int(value))
                elif key not in INT_FIELDS:
                    setattr(filters, key, value)
        else:
            filters = ResearchFilter(
                research_id=research.research_id,
                **updated_data.get("filters", {})
            )
            db.add(filters)


        await db.execute(delete(Message).where(Message.research_id == research.research_id))
        db.add_all([
            Message(research_id=research.research_id, **{"send_by": msg[0], "message_text": msg[1]})
            for msg in new_data["messages"]
        ])

        analysis_query = select(NetworkAnalysis).where(NetworkAnalysis.research_id == research.research_id)
        analysis_result = await db.execute(analysis_query)
        analysis = analysis_result.scalars().first()

        if analysis:
            analysis.nodes = new_data["nodes"]
            analysis.links = new_data["links"]
            analysis.is_connected = new_data["is_connected"]
        else:
            analysis = NetworkAnalysis(
                research_id=research.research_id,
                nodes=new_data["nodes"],
                links=new_data["links"],
                is_connected=new_data["is_connected"]
            )
            db.add(analysis)

        comparisons_query = select(Comparisons).where(Comparisons.research_id == research.research_id)
        comparisons_result = await db.execute(comparisons_query)
        comparisons = comparisons_result.scalars().all()

        await db.commit()

        research_entry = {
            **research.to_dict(),
            "filters": filters.to_dict() if filters else None,
            "analysis": analysis.to_dict() if analysis else None,
            "comparisons": [comp.to_dict() for comp in comparisons] if comparisons else []
        }

        return JSONResponse(
            content={
                "status": "success",
                "message": "Research updated successfully",
                "data": research_entry
            },
            status_code=200
        )

    except HTTPException as he:
        logger.error(f"Error updating research data: {str(he.detail)}")
        raise he
    except Exception as e:
        logger.error(f"Error updating research data: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")