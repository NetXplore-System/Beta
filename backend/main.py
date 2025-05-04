from fastapi import FastAPI, HTTPException, File, UploadFile, Query, Depends, Request, BackgroundTasks, Form  # type: ignore      
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from fastapi.security import OAuth2PasswordBearer  # type: ignore
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import community.community_louvain as community_louvain
# from community import best_partition, modularity  # type: ignore        
import networkx as nx  # type: ignore
import networkx.algorithms.community as nx_community  # type: ignore
from dotenv import load_dotenv
from jose import jwt, JWTError  # type: ignore
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession  # type: ignore
import bcrypt  # type: ignore
from sqlalchemy.future import select # type: ignore
from sqlalchemy import delete # type: ignore
from database import verify_connection
import asyncio 
import database
from typing import List, Optional
from models import User, Research, ResearchFilter, NetworkAnalysis, Message, Comparisons
from utils import extract_messages, anonymize_name, clean_filter_value
import uuid
from wikipedia import router as wikipedia_router

load_dotenv()
origins = os.getenv("ALLOWED_ORIGINS", "").split(",") 
 
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

UPLOAD_FOLDER = "./uploads/" 
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 


app = FastAPI()

# wikipedia
app.include_router(wikipedia_router)

@app.on_event("startup")
async def startup():
    await verify_connection()
    async with database.engine.begin() as conn:
        await conn.run_sync(database.Base.metadata.create_all)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserUpdate(BaseModel):
    name: str = Field(None)
    email: EmailStr = Field(None)
    password: str = Field(None)


class OAuthUser(BaseModel):
    name: str
    email: str
    avatar: str



def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Decodes JWT token to extract current user data.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"user_id": user_id}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")



@app.get("/protected")
async def protected_route(current_user: dict = Depends(get_current_user)):
    """
    Test protected route that requires authentication.
    """
    return {"message": f"Hello, user {current_user['user_id']}"}



@app.post("/api/auth/google")
async def google_auth(user: OAuthUser, db: AsyncSession = Depends(database.get_db)):
    """
    Authenticates user using Google OAuth.
    """
    async with db as session:
        # Check if user exists
        stmt = select(User).where(User.email == user.email)
        result = await session.execute(stmt)
        existing_user = result.scalars().first()

        if existing_user:
            token = create_access_token(data={"user_id": str(existing_user.user_id)})
            return {
                "access_token": token,
                "token_type": "bearer",
                "user": {
                    "id": str(existing_user.user_id),
                    "name": existing_user.name,
                    "email": existing_user.email,
                    "avatar": existing_user.avatar or "https://cdn-icons-png.flaticon.com/512/64/64572.png"
                },
            }

        # Create new user
        user_id = str(uuid4())
        new_user = User(
            user_id=user_id, 
            name=user.name, 
            email=user.email, 
            avatar=user.avatar,
            password=None  # OAuth users don't have passwords
        )
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)

        token = create_access_token(data={"user_id": str(new_user.user_id)})
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": str(new_user.user_id), 
                "name": new_user.name, 
                "email": new_user.email, 
                "avatar": new_user.avatar
            },
        }


@app.post("/register")
async def register_user(user: UserCreate, db: AsyncSession = Depends(database.get_db)):
    """Registers a new user."""
    async with db as session:
        # async with session.begin():
            # Check if email already exists
        stmt = select(User).where(User.email == user.email)
        result = await session.execute(stmt)
        existing_user = result.scalars().first()

        if existing_user:
            raise HTTPException(status_code=400, detail="Email already exists")

        # Hash the password
        hashed_password = bcrypt.hashpw(user.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        # Create and save the new user
        new_user = User(name=user.name, email=user.email, password=hashed_password)
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)

        token = create_access_token(data={"user_id": str(new_user.user_id)})

        return {
                "user":{
                    "id": str(new_user.user_id), 
                    "name": new_user.name, 
                    "email": new_user.email,
                    "avatar": new_user.avatar or "https://cdn-icons-png.flaticon.com/512/64/64572.png"
                },
                "access_token": token,
                "token_type": "bearer"
            }

  
@app.post("/login")  
async def login_user(user: UserLogin, db: AsyncSession = Depends(database.get_db)):
    """Logs in a user and returns a JWT token."""
    try:
        async with db as session:
            try:
                # Query the user
                stmt = select(User).where(User.email == user.email)
                result = await session.execute(stmt)
                db_user = result.scalars().first()

                if not db_user:
                    raise HTTPException(
                        status_code=401, 
                        detail="Invalid email or password"
                    )

                try:
                    # Verify password
                    if not bcrypt.checkpw(
                        user.password.encode("utf-8"), 
                        db_user.password.encode("utf-8")
                    ):
                        raise HTTPException(
                            status_code=401, 
                            detail="Invalid email or password"
                        )

                    # Generate token
                    token = create_access_token(data={"user_id": str(db_user.user_id)})

                    return {
                        "access_token": token,
                        "token_type": "bearer",
                        "user": {
                            "id": str(db_user.user_id),
                            "name": db_user.name,
                            "email": db_user.email,
                            "avatar": db_user.avatar or "https://cdn-icons-png.flaticon.com/512/64/64572.png",
                        },
                    }

                except Exception as password_error:
                    logger.error(f"Password verification error: {password_error}")
                    raise HTTPException(
                        status_code=500,
                        detail="Error verifying password"
                    )

            except HTTPException:
                raise  # Re-raise HTTP exceptions
            except Exception as query_error:
                logger.error(f"Database query error: {query_error}")
                raise HTTPException(
                    status_code=500,
                    detail="Error querying database"
                )

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during login"
        )


@app.get("/users")
async def get_all_users(db: AsyncSession = Depends(database.get_db)):
    """Fetches a list of all users."""
    async with db as session:
        result = await session.execute(select(User))
        users = result.scalars().all()
        return [{"id": str(user.user_id), "name": user.name, "email": user.email} for user in users]


@app.put("/users/{user_id}")
async def update_user(user_id: str, user_update: UserUpdate, db: AsyncSession = Depends(database.get_db)):
    """Updates a user's details."""
    async with db as session:
        # async with session.begin():
        user = await session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        update_data = user_update.dict(exclude_unset=True)
        if "password" in update_data:
            update_data["password"] = bcrypt.hashpw(update_data["password"].encode("utf-8"), bcrypt.gensalt()).decode(
                "utf-8")

        for key, value in update_data.items():
            setattr(user, key, value)

        await session.commit()
        await session.refresh(user)

        return {
            "id": str(user.user_id),
            "name": user.name,
            "email": user.email,
            "avatar": user.avatar or "https://cdn-icons-png.flaticon.com/512/64/64572.png"
        }


@app.delete("/users/{user_id}")
async def delete_user(user_id: str, db: AsyncSession = Depends(database.get_db)):
    """Deletes a user."""
    async with db as session:
        async with session.begin():
            user = await session.get(User, user_id)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            await session.delete(user)
            await session.commit()
            return {"message": "User deleted successfully"}



@app.post("/upload-avatar")
async def upload_avatar(
    file: UploadFile = File(...), 
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(database.get_db)
):
    """
    Uploads a new avatar for the authenticated user.
    """
    current_user = get_current_user(token)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    avatar_folder = os.path.join(UPLOAD_FOLDER, "avatars")
    os.makedirs(avatar_folder, exist_ok=True)
    avatar_filename = f"{current_user['user_id']}_{file.filename}"
    avatar_path = os.path.join(avatar_folder, avatar_filename)

    with open(avatar_path, "wb") as avatar_file:
        avatar_file.write(await file.read())

    avatar_url = f"/static/avatars/{avatar_filename}"
    
    async with db as session:
        user = await session.get(User, current_user["user_id"])
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        user.avatar = avatar_url
        await session.commit()
    
    return {"avatarUrl": avatar_url}



async def delayed_file_processing(filename: str):
    """Executes 120 seconds after file upload"""
    await asyncio.sleep(120)
    
    # Add your custom processing logic here
    print(f"Processing file {filename} after 2 minutes")
    # Example: Move file to permanent storage
    # Example: Run analysis and save results
    
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks
):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Add delayed task
        background_tasks.add_task(delayed_file_processing, file.filename)
        
        return JSONResponse(
            content={"message": "File uploaded successfully!", "filename": file.filename},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return JSONResponse(content={"message": f"File '{filename}' deleted successfully!"}, status_code=200)
        else:
            return JSONResponse(content={"error": f"File '{filename}' not found."}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)





@app.get("/analyze/network/{filename}")
async def analyze_network(
        filename: str,
        start_date: str = Query(None),
        start_time: str = Query(None),
        end_date: str = Query(None),
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
        anonymize: bool = Query(False)
):
    try:
        print(f"Analyzing file: {filename}, Anonymization: {anonymize}, Limit Type: {limit_type}")
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            return JSONResponse(content={"error": f"File '{filename}' not found."}, status_code=404)

        nodes = set()
        user_message_count = defaultdict(int)
        edges_counter = defaultdict(int)
        previous_sender = None
        anonymized_map = {}

        keyword_list = [kw.strip().lower() for kw in keywords.split(",")] if keywords else []
        selected_user_list = [user.strip().lower() for user in selected_users.split(",")] if selected_users else []

        start_datetime = None
        end_datetime = None

        if start_date and start_time:
            start_datetime = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M:%S")
        elif start_date:
            start_datetime = datetime.strptime(f"{start_date} 00:00:00", "%Y-%m-%d %H:%M:%S")

        if end_date and end_time:
            end_datetime = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M:%S")
        elif end_date:
            end_datetime = datetime.strptime(f"{end_date} 23:59:59", "%Y-%m-%d %H:%M:%S")

        print(f"🔹 Converted: start_datetime={start_datetime}, end_datetime={end_datetime}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        filtered_lines = []
        for line in lines:
            # print(f"🔹 Line: {line}")
            if line.startswith("[") and "]" in line:
                date_part = line.split("] ")[0].strip("[]")
                try:
                    current_datetime = datetime.strptime(date_part, "%d.%m.%Y, %H:%M:%S")
                except ValueError:
                    continue
                
                if "~" not in line:
                    # print(f"🔹 Line: {line}")
                    continue
                if "צירפת את" in line or "הצטרף/ה" in line or "צירף/ה" in line or "התמונה הושמטה" in line or "צורפת על ידי" in line or "הודעה זו נמחקה" in line or "הקבוצה נוצרה על ידי" in line:
                    continue

                if ((start_datetime and current_datetime >= start_datetime) or not start_datetime) and \
                        ((end_datetime and current_datetime <= end_datetime) or not end_datetime):
                    filtered_lines.append(line)
            else:
                if filtered_lines:
                    filtered_lines[-1] += line
                else:
                    filtered_lines.append(line)

        print(f"🔹 Found {len(filtered_lines)} messages in the date range.")

        if limit and limit_type == "first":
            selected_lines = filtered_lines[:limit]
        elif limit and limit_type == "last":
            selected_lines = filtered_lines[-limit:]
        else:
            selected_lines = filtered_lines

        print(f"🔹 Processing {len(selected_lines)} messages (Limit Type: {limit_type})")

        for line in selected_lines:
            try:
                if "omitted" in line or "omitted" in line:
                    continue

                if line.startswith("[") and "]" in line and ": " in line:
                    _, message_part = line.split("] ", 1)
                    parts = message_part.split(":", 1)
                    sender = parts[0].strip("~").replace("\u202a", "").strip()
                    message_content = parts[1].strip() if len(parts) > 1 else ""
                    # print(f"🔹 Sender: {sender}, Message: {message_content}")
                    message_length = len(message_content)
                    if (min_length and message_length < min_length) or (max_length and message_length > max_length):
                        continue

                    if username and sender.lower() != username.lower():
                        continue

                    if keywords and not any(kw in message_content.lower() for kw in keyword_list):
                        continue

                    user_message_count[sender] += 1

                    if sender:
                        if anonymize:
                            sender = anonymize_name(sender, anonymized_map)

                        nodes.add(sender)
                        if previous_sender and previous_sender != sender:
                            edge = tuple(sorted([previous_sender, sender]))
                            edges_counter[edge] += 1
                        previous_sender = sender
            except Exception as e:
                print(f"Error processing line: {line.strip()} - {e}")
                continue

        filtered_users = {
            user: count for user, count in user_message_count.items()
            if (not min_messages or count >= min_messages) and (not max_messages or count <= max_messages)
        }

        if active_users:
            sorted_users = sorted(filtered_users.items(), key=lambda x: x[1], reverse=True)[:active_users]
            filtered_users = dict(sorted_users)

        if selected_users:
            filtered_users = {user: count for user, count in filtered_users.items()
                              if user.lower() in selected_user_list}

        filtered_nodes = set(filtered_users.keys())
        if anonymize:
            filtered_nodes = {anonymize_name(node, anonymized_map) for node in filtered_nodes}

        G = nx.Graph()
        G.add_nodes_from(filtered_nodes)
        for (source, target), weight in edges_counter.items():
            if source in filtered_nodes and target in filtered_nodes:
                G.add_edge(source, target, weight=weight)

        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, weight="weight", normalized=True)
        if not nx.is_connected(G):
            print("Warning: The graph is not fully connected. Betweenness centrality might be inaccurate.")

        if nx.is_connected(G):
            closeness_centrality = nx.closeness_centrality(G)
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
            pagerank_centrality = nx.pagerank(G, alpha=0.85)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            G_subgraph = G.subgraph(largest_cc).copy()
            closeness_centrality = nx.closeness_centrality(G_subgraph)
            eigenvector_centrality = nx.eigenvector_centrality(G_subgraph, max_iter=1000)
            pagerank_centrality = nx.pagerank(G_subgraph, alpha=0.85)

        nodes_list = [
            { 
                "id": node,
                "messages": user_message_count.get(node, 0),
                "degree": round(degree_centrality.get(node, 0), 4),
                "betweenness": round(betweenness_centrality.get(node, 0), 4),
                "closeness": round(closeness_centrality.get(node, 0), 4),
                "eigenvector": round(eigenvector_centrality.get(node, 0), 4),
                "pagerank": round(pagerank_centrality.get(node, 0), 4),
            }
            for node in filtered_nodes
        ]

        links_list = []
        for edge, weight in edges_counter.items():
            source, target = edge

            if anonymize:
                source = anonymized_map.get(source, source)
                target = anonymized_map.get(target, target)

            if source in filtered_nodes and target in filtered_nodes:
                links_list.append({
                    "source": source,
                    "target": target,
                    "weight": weight
                })

        return JSONResponse(content={"nodes": nodes_list, "links": links_list}, status_code=200)
    except Exception as e:
        print("Error:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)



def apply_comparison_filters(network_data, node_filter, min_weight):
    """Filter network by node filter and minimum weight."""
    if not network_data or "nodes" not in network_data or "links" not in network_data:
        return network_data

    filtered_nodes = []
    if node_filter:
        filtered_nodes = [
            node for node in network_data["nodes"]
            if node_filter.lower() in node["id"].lower()
        ]
    else:
        filtered_nodes = network_data["nodes"]

    node_ids = {node["id"] for node in filtered_nodes}

    filtered_links = [
        link for link in network_data["links"]
        if (link["weight"] >= min_weight and
            (get_node_id(link["source"]) in node_ids) and
            (get_node_id(link["target"]) in node_ids))
    ]

    return {"nodes": filtered_nodes, "links": filtered_links}


def get_node_id(node_ref):
    """Get node ID whether it's a string or an object."""
    if isinstance(node_ref, dict) and "id" in node_ref:
        return node_ref["id"]
    return node_ref


def find_common_nodes(original_data, comparison_data):
    """Find common nodes between two networks."""
    original_ids = {node["id"] for node in original_data["nodes"]}
    comparison_ids = {node["id"] for node in comparison_data["nodes"]}
    return original_ids.intersection(comparison_ids)


def mark_common_nodes(network_data, common_node_ids):
    """Mark common nodes in a network."""
    for node in network_data["nodes"]:
        node["isCommon"] = node["id"] in common_node_ids
    return network_data


def get_network_metrics(original_data, comparison_data, metrics_list):
    """Calculate network metrics for comparison."""
    if not metrics_list:
        return {}

    metrics_names = [m.strip() for m in metrics_list.split(",")]
    results = {}

    results["node_count"] = {
        "original": len(original_data["nodes"]),
        "comparison": len(comparison_data["nodes"]),
        "difference": len(comparison_data["nodes"]) - len(original_data["nodes"]),
        "percent_change": (
            ((len(comparison_data["nodes"]) - len(original_data["nodes"])) / len(original_data["nodes"])) * 100
            if len(original_data["nodes"]) > 0 else 0
        )
    }

    results["link_count"] = {
        "original": len(original_data["links"]),
        "comparison": len(comparison_data["links"]),
        "difference": len(comparison_data["links"]) - len(original_data["links"]),
        "percent_change": (
            ((len(comparison_data["links"]) - len(original_data["links"])) / len(original_data["links"])) * 100
            if len(original_data["links"]) > 0 else 0
        )
    }

    return results


@app.get("/analyze/compare-networks")
async def analyze_network_comparison(
        original_filename: str = Query(...),
        comparison_filename: str = Query(...),
        start_date: str = Query(None),
        start_time: str = Query(None),
        end_date: str = Query(None),
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
        min_weight: int = Query(1),
        node_filter: str = Query(""),
        highlight_common: bool = Query(False),
        metrics: str = Query(None)
):
    try:
        print(f"Analyzing comparison between {original_filename} and {comparison_filename}")

        original_result = await analyze_network(
            original_filename, start_date, start_time, end_date, end_time,
            limit, limit_type, min_length, max_length, keywords,
            min_messages, max_messages, active_users, selected_users,
            username, anonymize
        )

        comparison_result = await analyze_network(
            comparison_filename, start_date, start_time, end_date, end_time,
            limit, limit_type, min_length, max_length, keywords,
            min_messages, max_messages, active_users, selected_users,
            username, anonymize
        )

        if hasattr(original_result, 'body'):
            original_data = json.loads(original_result.body)
        else:
            original_data = original_result

        if hasattr(comparison_result, 'body'):
            comparison_data = json.loads(comparison_result.body)
        else:
            comparison_data = comparison_result

        filtered_original = apply_comparison_filters(original_data, node_filter, min_weight)
        filtered_comparison = apply_comparison_filters(comparison_data, node_filter, min_weight)

        if highlight_common:
            common_nodes = find_common_nodes(filtered_original, filtered_comparison)
            mark_common_nodes(filtered_original, common_nodes)
            mark_common_nodes(filtered_comparison, common_nodes)

        return JSONResponse(content={
            "original": filtered_original,
            "comparison": filtered_comparison,
            "metrics": get_network_metrics(filtered_original, filtered_comparison, metrics)
        }, status_code=200)

    except Exception as e:
        print(f"Error in network comparison: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/analyze/communities/{filename}")
async def analyze_communities(
        filename: str,
        start_date: str = Query(None),
        start_time: str = Query(None),
        end_date: str = Query(None),
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
        algorithm: str = Query("louvain")
):
    try:
        network_result = await analyze_network(
            filename, start_date, start_time, end_date, end_time, limit, limit_type,
            min_length, max_length, keywords, min_messages, max_messages,
            active_users, selected_users, username, anonymize
        )

        if hasattr(network_result, 'body'):
            network_data = json.loads(network_result.body)
        else:
            network_data = network_result

        if "error" in network_data:
            return JSONResponse(content=network_data, status_code=400)

        G = nx.Graph()

        for node in network_data["nodes"]:
            G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})

        for link in network_data["links"]:
            source = link["source"]
            target = link["target"]
            weight = link.get("weight", 1)

            if isinstance(source, dict) and "id" in source:
                source = source["id"]
            if isinstance(target, dict) and "id" in target:
                target = target["id"]

            G.add_edge(source, target, weight=weight)

        communities = {}
        node_communities = {}

        if algorithm == "louvain":
            partition = community_louvain.best_partition(G)
            node_communities = partition

            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)

        elif algorithm == "girvan_newman":
            communities_iter = nx_community.girvan_newman(G)
            communities_list = list(next(communities_iter))

            for i, community in enumerate(communities_list):
                communities[i] = list(community)
                for node in community:
                    node_communities[node] = i

        elif algorithm == "greedy_modularity":
            communities_list = list(nx_community.greedy_modularity_communities(G))

            for i, community in enumerate(communities_list):
                communities[i] = list(community)
                for node in community:
                    node_communities[node] = i
        else:
            return JSONResponse(
                content={
                    "error": f"Unknown algorithm: {algorithm}. Supported: louvain, girvan_newman, greedy_modularity"},
                status_code=400
            )

        communities_list = [
            {
                "id": community_id,
                "size": len(nodes),
                "nodes": nodes,
                "avg_betweenness": sum(network_data["nodes"][i]["betweenness"]
                                       for i, node in enumerate(network_data["nodes"])
                                       if node["id"] in nodes) / len(nodes) if nodes else 0,
                "avg_pagerank": sum(network_data["nodes"][i]["pagerank"]
                                    for i, node in enumerate(network_data["nodes"])
                                    if node["id"] in nodes) / len(nodes) if nodes else 0,
            }
            for community_id, nodes in communities.items()
        ]

        communities_list.sort(key=lambda x: x["size"], reverse=True)

        for i, node in enumerate(network_data["nodes"]):
            node_id = node["id"]
            if node_id in node_communities:
                network_data["nodes"][i]["community"] = node_communities[node_id]

        return JSONResponse(content={
            "nodes": network_data["nodes"],
            "links": network_data["links"],
            "communities": communities_list,
            "node_communities": node_communities,
            "algorithm": algorithm,
            "num_communities": len(communities),
            "modularity": community_louvain.modularity(node_communities, G) if algorithm == "louvain" else None
        }, status_code=200)

    except Exception as e:
        print(f"Error in community detection: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

class CommunityAnalysisData(BaseModel):
    nodes: List[dict]
    links: List[dict]
    algorithm: str = Query("louvain")

@app.post("/history/analyze/communities") 
async def analyze_communities_history(
        data: CommunityAnalysisData
):
    try:    
        algorithm = data.algorithm
    
        G = nx.Graph()

        for node in data.nodes:
            G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})

        for link in data.links:
            source = link["source"]
            target = link["target"]
            weight = link.get("weight", 1)

            if isinstance(source, dict) and "id" in source:
                source = source["id"]
            if isinstance(target, dict) and "id" in target:
                target = target["id"]

            G.add_edge(source, target, weight=weight)

        communities = {}
        node_communities = {}

        if algorithm == "louvain":
            partition = community_louvain.best_partition(G)
            node_communities = partition

            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)

        elif algorithm == "girvan_newman":
            communities_iter = nx_community.girvan_newman(G)
            communities_list = list(next(communities_iter))

            for i, community in enumerate(communities_list):
                communities[i] = list(community)
                for node in community:
                    node_communities[node] = i

        elif algorithm == "greedy_modularity":
            communities_list = list(nx_community.greedy_modularity_communities(G))

            for i, community in enumerate(communities_list):
                communities[i] = list(community)
                for node in community:
                    node_communities[node] = i
        else:
            return JSONResponse(
                content={
                    "error": f"Unknown algorithm: {algorithm}. Supported: louvain, girvan_newman, greedy_modularity"},
                status_code=400
            )

        communities_list = [
            {
                "id": community_id,
                "size": len(nodes),
                "nodes": nodes,
                "avg_betweenness": sum(data.nodes[i]["betweenness"]
                                       for i, node in enumerate(data.nodes)
                                       if node["id"] in nodes) / len(nodes) if nodes else 0,
                "avg_pagerank": sum(data.nodes[i]["pagerank"]
                                    for i, node in enumerate(data.nodes)
                                    if node["id"] in nodes) / len(nodes) if nodes else 0,
            }
            for community_id, nodes in communities.items()
        ]

        communities_list.sort(key=lambda x: x["size"], reverse=True)

        for i, node in enumerate(data.nodes):
            node_id = node["id"]
            if node_id in node_communities:
                data.nodes[i]["community"] = node_communities[node_id]

        return JSONResponse(content={
            "nodes": data.nodes,
            "links": data.links,
            "communities": communities_list,
            "node_communities": node_communities,
            "algorithm": algorithm,
            "num_communities": len(communities),
            "modularity": community_louvain.modularity(node_communities, G) if algorithm == "louvain" else None
        }, status_code=200)

    except Exception as e:
        print(f"Error in community detection: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)




class NetworkAnalysisData(BaseModel):
    nodes: List[dict]
    links: List[dict]
    metric_name: Optional[str] = None

@app.post("/save-research")
async def save_research(
    file_name: str = Form(...),
    researcher_id: str = Form(...),
    research_name: str = Form(...),
    description: Optional[str] = Form(None),
    comparison: Optional[str] = Form(None),
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
    algorithm: str = Query("louvain"),
    # network_analysis: List[NetworkAnalysisData] = Form(...),
    db: AsyncSession = Depends(database.get_db)
):
    

    try:

        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        if not os.path.exists(file_path):
            return JSONResponse(content={"error": f"File '{file_name}' not found."}, status_code=404)
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        data = await extract_messages(lines, start_date, end_date, start_time, end_time, limit, limit_type, min_length, max_length, keywords, min_messages, max_messages, active_users, selected_users, username, anonymize)

        # print(f"🔹 Messages: {data['messages']}")
        # return JSONResponse(content={"data": data}, status_code=200)
        new_research = Research(
            research_name=research_name,
            description=description,
            user_id=researcher_id,
            platform=platform,
            # created_at=datetime.datetime.utcnow()
        )
        db.add(new_research)
        await db.commit()
        await db.refresh(new_research)

        for message in data["messages"]:
            new_message = Message(
                research_id=new_research.research_id,
                message_text=message[1],
                send_by=message[0],
                # created_at=datetime.datetime.utcnow()
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
            algorithm=algorithm
        )
        db.add(new_filter) 

        new_analysis = NetworkAnalysis(
            research_id=new_research.research_id,
            nodes=data['nodes'],
            links=data['links'],
            is_connected=data['is_connected'],
            metric_name=selected_metric
        )
        db.add(new_analysis)
        await db.commit()
        await db.refresh(new_analysis)

        # Handle comparison data if present
        if comparison:
            try:
                comparison_data = json.loads(comparison)
                if isinstance(comparison_data, list):
                    # Handle multiple comparisons
                    for comp_data in comparison_data:
                        print(f"🔹 Comparison Data: {comp_data}")
                        new_comparison = Comparisons(
                            research_id=new_research.research_id,
                            original_analysis=new_analysis.id,
                            nodes=comp_data.get('nodes', []),
                            links=comp_data.get('links', []),
                            is_connected=comp_data.get('is_connected', True)
                        )
                        db.add(new_comparison)
                elif isinstance(comparison_data, dict):
                    # Handle single comparison
                    new_comparison = Comparisons(
                        research_id=new_research.research_id,
                        original_analysis=new_analysis.id,
                        nodes=comparison_data.get('nodes', []),
                        links=comparison_data.get('links', []),
                        is_connected=comparison_data.get('is_connected', True)
                    )
                    db.add(new_comparison)
                
                await db.commit()
            except json.JSONDecodeError:
                print(f"Invalid comparison data format: {comparison}")
                # Continue without saving comparison data
                pass

        return {
            "message": "Data saved successfully", 
            "research_id": str(new_research.research_id)
        }

    except Exception as e:
        print(f"Error saving data: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving data: {str(e)}")


@app.get("/history/{user_id}")
async def get_user_history(
    user_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(database.get_db)
):
    """Get all research history for a user including filters, analysis and comparisons."""
    try:
        # Convert string user_id to UUID
        user_uuid = uuid.UUID(user_id)

        # Ensure user_id matches current user's ID
        if str(user_uuid) != current_user["user_id"]:
            raise HTTPException(
                status_code=403,
                detail="Access forbidden: You can only view your own research history"
            )
        
        # Query all research entries for the user
        query = select(Research).where(Research.user_id == user_uuid)
        result = await db.execute(query)
        researches = result.scalars().all()
        
        # Prepare response data
        history = []
        
        for research in researches:
            # Get research filters
            filter_query = select(ResearchFilter).where(
                ResearchFilter.research_id == research.research_id
            )
            filter_result = await db.execute(filter_query)
            filters = filter_result.scalars().first()
            
            # Get network analysis
            analysis_query = select(NetworkAnalysis).where(
                NetworkAnalysis.research_id == research.research_id
            )
            analysis_result = await db.execute(analysis_query)
            analysis = analysis_result.scalars().first()
            
            # Get comparisons if they exist
            comparison_query = select(Comparisons).where(
                Comparisons.research_id == research.research_id
            )
            comparison_result = await db.execute(comparison_query)
            comparisons = comparison_result.scalars().all()
            
            # Build complete research entry
            research_entry = {
                **research.to_dict(),
                "filters": filters.to_dict() if filters else None,
                "analysis": analysis.to_dict() if analysis else None,
                "comparisons": [comp.to_dict() for comp in comparisons] if comparisons else []
            }
            
            history.append(research_entry) 
        
        return JSONResponse(
            content={
                "status": "success",
                "history": history
            },
            status_code=200
        )
            
    except ValueError as ve:
        logger.error(f"Invalid UUID format: {ve}")
        raise HTTPException(
            status_code=400,
            detail="Invalid user ID format"
        )
    except Exception as e:
        logger.error(f"Error fetching user history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching history: {str(e)}"
        )

@app.delete("/research/{research_id}")
async def delete_research(
    research_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(database.get_db)
):
    """Delete a research and all its related data"""
    try:
        # Verify research exists and belongs to current user
        research = await db.get(Research, research_id)
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        if str(research.user_id) != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Not authorized to delete this research")

        # Delete comparisons first
        await db.execute(
            delete(Comparisons)
            .where(Comparisons.research_id == research_id)
        )

        # Delete network analysis
        await db.execute(
            delete(NetworkAnalysis)
            .where(NetworkAnalysis.research_id == research_id)
        )

        # Delete research filters
        await db.execute(
            delete(ResearchFilter)
            .where(ResearchFilter.research_id == research_id)
        )

        # Delete messages
        await db.execute(
            delete(Message)
            .where(Message.research_id == research_id)
        )

        # Finally delete the research itself
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
        raise he
    except Exception as e:
        logger.error(f"Error deleting research: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting research: {str(e)}"
        )
    
  

@app.get("/history/analyze/compare")
async def analyze_network_comparison_history(
        research_id: str = Query(...),
        min_weight: int = Query(1),
        node_filter: str = Query(""),
        highlight_common: bool = Query(False),
        metrics: str = Query(None),
        comparison_index: int = Query(0),
        db: AsyncSession = Depends(database.get_db)
):
    try:
       
        research = await db.get(Research, research_id)
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        
        
        analysis_query = select(NetworkAnalysis).where(NetworkAnalysis.research_id == research_id)
        original_result = await db.execute(analysis_query)
        original_data = original_result.scalars().first()

        # Extract the comparison data
        comparison_query = select(Comparisons).where(Comparisons.research_id == research_id)
        comparison_result = await db.execute(comparison_query)
        comparison_data = comparison_result.scalars().all()

        # Convert to dictionary if necessary
        if original_data:
            original_data = original_data.to_dict()
        if comparison_index < 0 or comparison_index >= len(comparison_data):
            raise HTTPException(status_code=404, detail="Comparison index out of range")

        specific_comparison = comparison_data[comparison_index].to_dict()

        filtered_original = apply_comparison_filters(original_data, node_filter, min_weight)
        filtered_comparison = apply_comparison_filters(specific_comparison, node_filter, min_weight)

        if highlight_common:
            common_nodes = find_common_nodes(filtered_original, filtered_comparison)
            mark_common_nodes(filtered_original, common_nodes)
            mark_common_nodes(filtered_comparison, common_nodes)

        return JSONResponse(content={
            "original": filtered_original,
            "comparison": filtered_comparison,
            "metrics": get_network_metrics(filtered_original, filtered_comparison, metrics)
        }, status_code=200)

    except Exception as e:
        print(f"Error in network comparison: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)



@app.put("/research/{research_id}")
async def update_research_data(
    research_id: str,
    updated_data: dict,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(database.get_db)
):
    try:
        # --- 1. Validation & permissions ---
        research = await db.get(Research, research_id)
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        if str(research.user_id) != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Not authorized")

        # --- 2. File existence check ---
        file_name = updated_data.get("file_name")
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        if not os.path.exists(file_path):
            return JSONResponse(
                content={"error": f"File '{file_name}' not found."},
                status_code=404
            )

        # --- 3. Extract messages and network data ---
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        filters_data = {
            **updated_data.get("filters", {}), 
            "limit": int(updated_data.get("filters", {}).get("message_limit") or 0),
            "min_length": int(updated_data.get("filters", {}).get("min_message_length") or 0),
            "max_length": int(updated_data.get("filters", {}).get("max_message_length") or 0),
            "active_users": int(updated_data.get("filters", {}).get("top_active_users") or 0),
            "min_messages": int(updated_data.get("filters", {}).get("min_messages") or 0),
            "max_messages": int(updated_data.get("filters", {}).get("max_messages") or 0),
            "username": updated_data.get("filters", {}).get("filter_by_username"),
        }

        filters_data.pop("message_limit")
        filters_data.pop("min_message_length")
        filters_data.pop("max_message_length")
        filters_data.pop("top_active_users")
        filters_data.pop("filter_by_username")
        filters_data.pop("algorithm")

        new_data = await extract_messages(lines, **filters_data)

        # --- 4. Update Research (name, description, file_name) ---
        research.research_name = updated_data.get("research_name", research.research_name)
        research.description = updated_data.get("description", research.description)
        research.file_name = file_name or research.file_name

        # --- 5. Update or Create Filters ---
        filter_query = select(ResearchFilter).where(ResearchFilter.research_id == research.research_id)
        filter_result = await db.execute(filter_query)
        filters = filter_result.scalars().first()

        INT_FIELDS = [
            "message_limit", "min_message_length", "max_message_length",
            "min_messages", "max_messages", "top_active_users"
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


        # --- 6. Replace all Messages ---
        await db.execute(delete(Message).where(Message.research_id == research.research_id))
        db.add_all([
            Message(research_id=research.research_id, **{"send_by": msg[0], "message_text": msg[1]})
            for msg in new_data["messages"]
        ])

        # --- 7. Update or Create Analysis ---
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

        # --- 8. Get Comparisons (Read only) ---
        comparisons_query = select(Comparisons).where(Comparisons.research_id == research.research_id)
        comparisons_result = await db.execute(comparisons_query)
        comparisons = comparisons_result.scalars().all()

        await db.commit()

        # --- 9. Build final response ---
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
        raise he
    except Exception as e:
        logger.error(f"Error updating research data: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


