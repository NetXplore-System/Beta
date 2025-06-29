import copy
from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from collections import defaultdict
from bs4 import Tag
import requests
from bs4 import BeautifulSoup
import logging
import json
import traceback
from typing import Optional
from datetime import datetime, time
import re
import unicodedata as ud
import networkx as nx
from community import community_louvain
from networkx.algorithms import community as nx_community
from utils import (
    normalize_links_by_target,
    calculate_sequential_weights_from_comments,
)


router = APIRouter()
logger = logging.getLogger("wikipedia")
logging.basicConfig(level=logging.DEBUG) 
logger.setLevel(logging.DEBUG)



@router.get("/analyze/wikipedia/{section_title}")
async def analyze_wikipedia_endpoint(
    section_title: str,
    limit: Optional[int] = Query(None),
    limit_type: str = Query("first"),
    min_length: Optional[int] = Query(None),
    max_length: Optional[int] = Query(None),
    min_messages: Optional[int] = Query(None),
    max_messages: Optional[int] = Query(None),
    active_users: Optional[int] = Query(None),
    selected_users: Optional[str] = Query(None),
    username: Optional[str] = Query(None),
    anonymize: bool = Query(False),
    directed: bool = Query(False),
    use_history: bool = Query(False),
    normalize: bool = Query(False),
    include_messages: bool = Query(False),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    history_length: int = Query(3),
    message_weights: Optional[str] = Query(None),
    keywords: Optional[str] = Query(None),
    content_filter: Optional[str] = Query(None),
):
    return await analyze_wikipedia(
        section_title, limit, limit_type, min_length, max_length, min_messages, max_messages,
        active_users, selected_users, username, anonymize, directed, use_history, normalize,
        include_messages, start_date, end_date, start_time, end_time, history_length,
        message_weights, keywords, content_filter
    )


    
@router.get("/analyze/wikipedia-communities/{section_title}")
async def analyze_wikipedia_communities(
    section_title: str,
    limit: Optional[int] = Query(None),
    limit_type: str = Query("first"),
    min_length: Optional[int] = Query(None),
    max_length: Optional[int] = Query(None),
    min_messages: Optional[int] = Query(None),
    max_messages: Optional[int] = Query(None),
    active_users: Optional[int] = Query(None),
    selected_users: Optional[str] = Query(None),
    username: Optional[str] = Query(None),
    anonymize: bool = Query(False),
    directed: bool = Query(False),
    use_history: bool = Query(False),
    normalize: bool = Query(False),
    include_messages: bool = Query(False),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    history_length: int = Query(3),
    message_weights: Optional[str] = Query(None),
    keywords: Optional[str] = Query(None),
    content_filter: Optional[str] = Query(None),
    algorithm: str = Query("louvain"),
):
    try:
        graph_data = await analyze_wikipedia(
            section_title, limit, limit_type, min_length, max_length, min_messages, max_messages, active_users, selected_users, username, 
            anonymize, directed, use_history, normalize, include_messages, start_date, end_date, start_time, end_time, history_length,
            message_weights, keywords, content_filter
        )
        graph_data = json.loads(graph_data.body)
        G = nx.Graph()
        for node in graph_data["nodes"]:
            G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})

        for link in graph_data["links"]:
            G.add_edge(link["source"], link["target"], weight=link.get("weight", 1))

        communities = {}
        node_communities = {}

        if algorithm == "louvain":
            partition = community_louvain.best_partition(G)
            node_communities = partition
            for node, cid in partition.items():
                communities.setdefault(cid, []).append(node)

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
            raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algorithm}")

        for node in graph_data["nodes"]:
            if node["id"] in node_communities:
                node["community"] = node_communities[node["id"]]

        communities_list = []
        for cid, nodes in communities.items():
            comm_nodes = [n for n in graph_data["nodes"] if n["id"] in nodes]
            avg_betweenness = sum(n["betweenness"] for n in comm_nodes) / len(comm_nodes) if comm_nodes else 0
            avg_pagerank = sum(n["pagerank"] for n in comm_nodes) / len(comm_nodes) if comm_nodes else 0
            avg_messages = sum(n["messages"] for n in comm_nodes) / len(comm_nodes) if comm_nodes else 0

            communities_list.append({
                "id": cid,
                "size": len(nodes),
                "nodes": nodes,
                "avg_betweenness": round(avg_betweenness, 4),
                "avg_pagerank": round(avg_pagerank, 4),
                "avg_messages": round(avg_messages, 2),
            })

        communities_list.sort(key=lambda x: x["size"], reverse=True)
        modularity = community_louvain.modularity(node_communities, G) if algorithm == "louvain" else None

        return JSONResponse(content= {
            "nodes": graph_data["nodes"],
            "links": graph_data["links"],
            "communities": communities_list,
            "node_communities": node_communities,
            "algorithm": algorithm,
            "num_communities": len(communities),
            "modularity": round(modularity,4) if modularity else None, 
            "is_connected": graph_data.get("is_connected", False)}, 
            status_code=200)
    except Exception as e:
        logger.error(f"Error in analyze_wikipedia_communities: {e}")    
        raise HTTPException(detail=str(e), status_code=500)




@router.post("/fetch-wikipedia-data")
async def fetch_wikipedia_data(request: Request):
    try:
        data = await request.json()
        url = data.get("url")
        if not url:
            logger.error("URL is required")
            return {"message": "URL is required"}
        data = save_wikipedia_data(url)
        if data:
            content = [{"title": section["title"], "len_comments": section["len_comments"]} for section in data]
            logger.info(f"Wikipedia data saved successfully for url: {url}")
            return JSONResponse(content= {"message": "Wikipedia data saved successfully", "data": content}, status_code=200)
        else:
            logger.error("Failed to save wikipedia data")
            raise HTTPException(content= {"message": "Failed to save wikipedia data"}, status_code=500)
    
    except Exception as e:
        logger.error(f"Error in fetch wikipedia: {e}")
        raise HTTPException(detail=str(e), status_code=500)



async def analyze_wikipedia(
    section_title: str,
    limit: Optional[int] = None,
    limit_type: str = "first", 
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    min_messages: Optional[int] = None,
    max_messages: Optional[int] = None,
    active_users: Optional[int] = None,
    selected_users: Optional[str] = None,
    username: Optional[str] = None,
    anonymize: bool = False,
    directed: bool = False,
    use_history: bool = False,
    normalize: bool = False,
    include_messages: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    history_length: int = 3,
    message_weights: Optional[str] = None,
    keywords: Optional[str] = None,
    content_filter: Optional[str] = None,
):
    try:
        username = str(username) if username is not None else None
        keywords = str(keywords) if keywords is not None else None  
        content_filter = str(content_filter) if content_filter is not None else None
        
        logger.info(f"Date filters: start_date={start_date}, end_date={end_date}, start_time={start_time}, end_time={end_time}")
        
        with open("uploads/wikipedia_data.json", "r", encoding="utf-8") as f:
            all_data = json.load(f)
        
        data = None
        for section in all_data:
            if section.get("title") == section_title:
                data = section
                break
        
        if data is None:
            raise HTTPException(detail=f"Section '{section_title}' not found in wikipedia_data.json", status_code=404)
        
        comments = data.get("comments", [])
        filtered_comments = comments
        count_msg_per_user = defaultdict(int)
        usernames = set()
        edges_counter = defaultdict(int)
        is_connected = False
        
        if start_date and end_date:
            filtered_comments = filter_comments_by_date(filtered_comments, start_date, end_date, start_time, end_time)
        
        if min_length:
            filtered_comments = [c for c in filtered_comments if len(c.get("text", "")) >= min_length]
            
        
        if max_length:
            filtered_comments = [c for c in filtered_comments if len(c.get("text", "")) <= max_length]
            print(f"Filtered to {len(filtered_comments)} comments with max length {max_length}")
        
        if keywords:
            keyword_list = [ud.normalize("NFKC", k.strip()) for k in keywords.split(",")]
            filtered_comments = [
                c for c in filtered_comments 
                if any(keyword in ud.normalize("NFKC", c.get("text", "")) for keyword in keyword_list)
            ]
        
        if username:
            filtered_comments = [c for c in filtered_comments if ud.normalize("NFC", c.get("writer_name", "")) == ud.normalize("NFC", username)]
        
        if content_filter:
            filtered_comments = [
                c for c in filtered_comments 
                if ud.normalize("NFC", content_filter) in ud.normalize("NFC", c.get("text", ""))
            ]
        
        if directed and use_history:
            if limit:
                if limit_type == "last":
                    filtered_comments = filtered_comments[-limit:]
                else:
                    filtered_comments = filtered_comments[:limit]
            else:
                if limit_type == "last":
                    filtered_comments = filtered_comments[::-1]
                else:
                    filtered_comments = filtered_comments
        else:
            if limit:
                if limit_type == "last":
                    filtered_comments = filtered_comments[-limit:][::-1]
                else:
                    filtered_comments = filtered_comments[:limit]
            else:
                if limit_type == "last":
                    filtered_comments = filtered_comments[::-1]
                else:
                    filtered_comments = filtered_comments
                
        for c in filtered_comments:
            writer_name = c["writer_name"]
            count_msg_per_user[writer_name] = count_msg_per_user.get(writer_name, 0) + 1
            
        if anonymize:
            anonymized_comments = {}
            for c in count_msg_per_user:
                anonymized_comments[c] = f"User_{len(anonymized_comments) + 1}"
            
            for c in filtered_comments:
                c["writer_name"] = anonymized_comments[c["writer_name"]]
                if c["reply_to"] and c["reply_to"] in anonymized_comments:
                    c["reply_to"] = anonymized_comments[c["reply_to"]]
                elif c["reply_to"]:
                    if c["reply_to"] not in anonymized_comments:
                        anonymized_comments[c["reply_to"]] = f"User_{len(anonymized_comments) + 1}"
                    c["reply_to"] = anonymized_comments[c["reply_to"]]
            
            new_count_msg_per_user = {}
            for real_name, count in count_msg_per_user.items():
                anonymous_name = anonymized_comments[real_name]
                new_count_msg_per_user[anonymous_name] = count
            count_msg_per_user = new_count_msg_per_user
            
                

        if min_messages or max_messages or active_users or selected_users:
        
            filtered_users = {u: c for u, c in count_msg_per_user.items()
                            if (not min_messages or min_messages == '' or c >= int(min_messages)) and
                                (not max_messages or max_messages == '' or c <= int(max_messages))}

            if active_users and active_users != '':
                sorted_users = sorted(filtered_users.items(), key=lambda x: x[1], reverse=True)
                filtered_users = dict(sorted_users[:int(active_users)])
                print(f"Filtered to top {active_users} active users")

            if selected_users:
                selected_set = set([u.strip().lower() for u in selected_users.split(",")])
                filtered_users = {u: c for u, c in filtered_users.items() if u.lower() in selected_set}
                print(f"Filtered to selected users: {len(filtered_users)}")

            usernames = set(filtered_users.keys())
        
        if directed and use_history:
            parsed_weights = json.loads(message_weights)
            history_n = int(history_length) if history_length else 3
            edges = calculate_sequential_weights_from_comments(filtered_comments, n_prev=history_n, message_weights=parsed_weights)
            for (source, target), weight in edges.items():
                edge = (source, target)
                edges_counter[edge] += weight
        else:
            for c in filtered_comments:
                if c["writer_name"] != c["reply_to"] and c["reply_to"] is not None:
                    edge = (c["writer_name"], c["reply_to"]) if directed else tuple(sorted([c["writer_name"], c["reply_to"]]))
                    edges_counter[edge] += 1
        
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(usernames)
        
        print(f"usernames: {usernames}")

        for edge, weight in edges_counter.items():
            if edge[0] in usernames and edge[1] in usernames:
                G.add_edge(edge[0], edge[1], weight=weight)

        print(f"Graph built: {len(G.nodes())} nodes, {len(G.edges())} edges")

        if directed:
            is_connected = nx.is_weakly_connected(G) if len(G.nodes()) > 0 else False
        else:
            is_connected = nx.is_connected(G) if len(G.nodes()) > 0 else False


        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G, weight="weight")
            
            if is_connected and len(G.nodes()) > 1:
                closeness_centrality = nx.closeness_centrality(G)
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
                pagerank = nx.pagerank(G)
            else:
                if len(G.nodes()) > 1:
                    if directed:
                        components = list(nx.weakly_connected_components(G))
                    else:
                        components = list(nx.connected_components(G))
                    
                    if components:
                        largest_cc = max(components, key=len)
                        subgraph = G.subgraph(largest_cc).copy()
                        
                        closeness_centrality = nx.closeness_centrality(subgraph)
                        eigenvector_centrality = nx.eigenvector_centrality(subgraph, max_iter=1000)
                        pagerank = nx.pagerank(subgraph)
                        
                        for node in G.nodes():
                            if node not in largest_cc:
                                closeness_centrality[node] = 0.0
                                eigenvector_centrality[node] = 0.0
                                pagerank[node] = 0.0
                    else:
                        closeness_centrality = {node: 0.0 for node in G.nodes()}
                        eigenvector_centrality = {node: 0.0 for node in G.nodes()}
                        pagerank = {node: 1.0/len(G.nodes()) for node in G.nodes()}
                else:
                    closeness_centrality = {}
                    eigenvector_centrality = {}
                    pagerank = {}
                    
        except Exception as e:
            logger.error(f"Error calculating centrality measures: {e}")
            degree_centrality = {}
            betweenness_centrality = {}
            closeness_centrality = {}
            eigenvector_centrality = {}
            pagerank = {}
            

        nodes_list = [
            {
                "id": user,
                "name": user,
                "group": 1,
                "messages": count_msg_per_user.get(user, 0),
                "degree": round(degree_centrality.get(user, 0), 4),
                "betweenness": round(betweenness_centrality.get(user, 0), 4),
                "closeness": round(closeness_centrality.get(user, 0), 4),
                "eigenvector": round(eigenvector_centrality.get(user, 0), 4),
                "pagerank": round(pagerank.get(user, 0), 4)
            }
            for user in usernames
        ]

        links_list = [
            {"source": a, "target": b, "weight": round(w, 3)}
            for (a, b), w in edges_counter.items()
            if a in usernames and b in usernames
        ]
        
        if normalize and directed and use_history:
            print("Normalizing link weights by target...")
            links_list = normalize_links_by_target(links_list)
            debug_check_target_weights(links_list)
            
       
        messages = [comment.get('text', '') for comment in filtered_comments] if include_messages else None
        
        
        logger.info(f"Filtered {len(comments)} comments to {len(filtered_comments)} comments, {len(nodes_list)} nodes")
        
        return JSONResponse(content={"messages": messages, "nodes": nodes_list, "links": links_list, "is_connected": is_connected}, status_code=200)
        
    except Exception as e:
        logger.error(f"Error in analyze_wikipedia: {e}")    
        raise HTTPException(detail=str(e), status_code=500)
 
def debug_check_target_weights(links):
    target_totals = defaultdict(float)
    for link in links:
        target_totals[link["target"]] += link["weight"]

    for target, total in target_totals.items():
        print(f"Target: {target}, Total Weight: {round(total, 4)}")


names_writers = {}

def process_wiki_talk_page(url):
    response = requests.get(url)
    html_content = response.text

    soup = BeautifulSoup(html_content, 'html.parser')

    content_div = soup.find("div", id="mw-content-text")
    if not content_div:
        logger.warning("Could not find any suitable content div")
        return []
    
    child = content_div.find("div", class_="mw-parser-output")
    if not child:
        logger.warning("Could not find any suitable child div")
        return []

    all_divs = child.find_all("div", class_="mw-heading mw-heading2 ext-discussiontools-init-section")
    sections = []
    print(f"all_divs: {len(all_divs)}")
    
    for i, div in enumerate(all_divs):
        h2_element = div.find("h2")
        title = None
        sum_comments = None
        if not h2_element:
            logger.warning(f"div without h2 element: {div}")
        else:
            title = h2_element.get_text(strip=True)
        sum_comments_element = div.find("span", class_="ext-discussiontools-init-section-commentCountLabel") 
        if not sum_comments_element:
            logger.warning(f"section '{title}' without comment count")
        else:
            sum_comments = sum_comments_element.get_text(strip=True)
        
        next_sibling = div.find_next_sibling()
        
        data = []
        
        while next_sibling and not (next_sibling.name == "div" and "ext-discussiontools-init-section" in next_sibling.get("class", [])):
            if next_sibling.name == "div" and "ext-discussiontools-init-section" in next_sibling.get("class", []):
                print(f"next_sibling: {next_sibling.get('class', [])}")
            if next_sibling.name == "p":

                comment_data, siblings_traversed = extract_comment_data(next_sibling)
                
                if comment_data:
                    data.append(comment_data)
                    
                if siblings_traversed > 0:
                    for _ in range(siblings_traversed):
                        if next_sibling:
                            next_sibling = next_sibling.find_next_sibling()
                
            elif next_sibling.name == "dl":
                dd_elements = next_sibling.find_all("dd", recursive=False)
                index_to_skip = []
                
                for j, dd in enumerate(dd_elements):
                    if j not in index_to_skip:
                        comment_data, siblings_traversed = extract_comment_data(dd) 
                        if comment_data:
                            data.append(comment_data)
                            
                        if siblings_traversed > 0:
                            for skip_offset in range(1, siblings_traversed + 1):
                                index_to_skip.append(j + skip_offset)
                        
                    nested_dl = dd.find("dl")
                    if nested_dl:
                        process_nested_replies(nested_dl, data)
                        
            elif next_sibling.name == "ol":
                dd_elements = next_sibling.find_all("li", recursive=False)
                index_to_skip = []
                
                for j, dd in enumerate(dd_elements):
                    if j not in index_to_skip:
                        comment_data, siblings_traversed = extract_comment_data(dd) 
                        if comment_data:
                            data.append(comment_data)
                            
                        if siblings_traversed > 0:
                            for skip_offset in range(1, siblings_traversed + 1):
                                index_to_skip.append(j + skip_offset)
                        
                    nested_dl = dd.find("dl")
                    if nested_dl:
                        process_nested_replies(nested_dl, data)
            
            next_sibling = next_sibling.find_next_sibling()
                
        sections.append({
            "title": title if title else "Unknown",
            "sum_comments": sum_comments if sum_comments else "Unknown",
            "comments": data if data else [],
            "len_comments": len(data) if data else 0
        })
                
    return sections



def process_nested_replies(nested_dl, data):

    dd_elements = nested_dl.find_all("dd", recursive=False)
    index_to_skip = []
    
    for i, dd in enumerate(dd_elements):
        if i not in index_to_skip:
            comment_data, siblings_traversed = extract_comment_data(dd)
            if comment_data:
                data.append(comment_data)
            
            if siblings_traversed > 0:
                for skip_offset in range(1, siblings_traversed + 1):
                    index_to_skip.append(i + skip_offset)
                    
        deeper_nested_dl = dd.find("dl")
        if deeper_nested_dl:
            process_nested_replies(deeper_nested_dl, data)
                
       
def extract_comment_data(next_sibling):
    
    try:
        comment_element = get_direct_element(next_sibling)
        
        if comment_element is None:
            logger.error(f"next_sibling type: {type(next_sibling)}")
            if hasattr(next_sibling, 'name'):
                logger.error(f"next_sibling name: {next_sibling.name}")
            return None, 0
            
        comment_text = None
        spans = comment_element.find_all("span")
        writer_name = None
        reply_to = None
        timestamp = None
        end_span = None
        siblings_traversed = 0
        
        for span in spans:
            if span.get("data-mw-comment-end"):
                end_span = span
                break
        
        if end_span:
            sender, reply_to_from_end, sender_timestamp, reply_to_timestamp = extract_sender_reply_and_timestamp(end_span.get("data-mw-comment-end"))
            reply_to = reply_to_from_end
            timestamp = sender_timestamp
            user_links = comment_element.find_all("a", href=True)
            writer_name = get_writer_name_from_links(user_links)
            if writer_name:
                names_writers[sender] = writer_name
            else:
                print(f"writer_name is None, sender: {sender}")
            comment_text = get_clean_text(comment_element, writer_name if writer_name else sender)
            writer_name = sender
            
        else:
            comment_text = next_sibling.get_text(strip=True)
            current_sibling = next_sibling

            while current_sibling:
                siblings_traversed += 1
                current_sibling = current_sibling.find_next_sibling()
                current_sibling_direct = get_direct_element(current_sibling)
                
                if not current_sibling:
                    break
                
                sibling_text = current_sibling_direct.get_text(strip=True)
                end_span = None
                spans_sibling = current_sibling_direct.find_all("span")
                for span in spans_sibling:
                    if span.get("data-mw-comment-end"):
                        end_span = span
                        break
                if end_span:
                    sender, reply_to_from_end, sender_timestamp, reply_to_timestamp = extract_sender_reply_and_timestamp(end_span.get("data-mw-comment-end"))
                    reply_to = reply_to_from_end
                    timestamp = sender_timestamp
                    user_links = current_sibling_direct.find_all("a", href=True)
                    writer_name = get_writer_name_from_links(user_links)
                    if writer_name:
                        names_writers[sender] = writer_name
                    else:
                        print(f"writer_name is None, sender: {sender}")
                    comment_text += " " + get_clean_text(current_sibling_direct, writer_name if writer_name else sender)
                    writer_name = sender
                    break
                else:
                    comment_text += " " + sibling_text
            
        
        
        if writer_name and timestamp:
            
            comment_data = {
                "reply_to": reply_to,
                "writer_name": writer_name,
                "text": comment_text,
                "timestamp": timestamp,
                "reply_to_timestamp": reply_to_timestamp if reply_to_timestamp and (reply_to_timestamp.isdigit() or ("Z" in reply_to_timestamp and "T" in reply_to_timestamp)) else None
            }
            
            return comment_data, siblings_traversed 
        else:
            return None, siblings_traversed
    except Exception as e:
        logger.error(f"Error in extract_comment_data:")
        logger.error(f"Exception: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"next_sibling: {get_direct_element(next_sibling)}")
        logger.error(f"next_sibling type: {type(next_sibling)}")
        if hasattr(next_sibling, 'name'):
            logger.error(f"next_sibling name: {next_sibling.name}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, 0
    
def get_clean_text(comment_element: Tag, writer_name: str = None) -> str:
   
    if not comment_element:
        return ""
    
    copy_comment_element = copy.copy(comment_element)
    
    sig_span = copy_comment_element.find("span", attrs={"data-mw-comment-sig": True})
    if sig_span:
        for sibling in list(sig_span.next_siblings):
            if hasattr(sibling, 'decompose'):
                sibling.decompose()
    
    text = copy_comment_element.get_text(strip=True)
    
    text = clean_unicode_formatting(text)
    
    if not writer_name or writer_name == "":
        return text.strip()

    original = text = text.strip()

    pattern = re.compile(rf"(.*?)\b{re.escape(writer_name)}\b.*$", re.DOTALL)

    match = pattern.match(text)
    if match:
        clean = match.group(1).rstrip()
        return clean

    print(f"Writer name '{writer_name}' not found at end of comment. Returning original text.")
    
    return original


def clean_unicode_formatting(text: str) -> str:
    
    if not text:
        return ""
    
    formatting_chars = [  '\u200F', '\u200E', '\u200D', '\u200C', '\u202A', '\u202B', '\u202C', '\u202D', '\u202E', '\u2066', '\u2067', '\u2068', '\u2069', '\uFEFF', '\u00AD']
    
    for char in formatting_chars:
        text = text.replace(char, '')
    
    return text

def get_writer_name_from_links(user_links):
   
    if not user_links:
        return None
    
    def extract_username(link):
        if not link:
            return None
        title = link.get("title", "")
        if title.startswith("משתמש") or title.startswith("User"):
            return link.get_text(strip=True)
        return None
    
    if len(user_links) > 3:
        username = extract_username(user_links[-3])
        if username:
            return username
    
    for link in user_links:
        username = extract_username(link)
        if username:
            return username
    
    return None

def extract_sender_reply_and_timestamp(comment_end_value):
    
    if not comment_end_value or not comment_end_value.startswith("c-"):
        return None, None, None, None
    
    parts = comment_end_value[2:].split("-")
    
    if "T" and "Z" in comment_end_value:
        # print(f"parts: {len(parts)}")
        # print(f"parts: {parts}")
        
        if len(parts) == 5:
            sender = clean_unicode_formatting(parts[0].replace("_", " "))
            sender_timestamp = get_clean_timestamp(parts[1] + parts[2] + parts[3])
            return sender, None, sender_timestamp, None
        elif len(parts) == 6:
            sender = clean_unicode_formatting(parts[0].replace("_", " "))
            sender_timestamp = get_clean_timestamp(parts[1])
            reply_to = clean_unicode_formatting(parts[2].replace("_", " "))
            reply_to_timestamp = get_clean_timestamp(parts[3] + parts[4] + parts[5])
            return sender, reply_to, sender_timestamp, reply_to_timestamp
        elif len(parts) == 8:
            sender = clean_unicode_formatting(parts[0].replace("_", " "))
            sender_timestamp = get_clean_timestamp(parts[1] + parts[2] + parts[3])
            reply_to = clean_unicode_formatting(parts[4].replace("_", " "))
            reply_to_timestamp = get_clean_timestamp(parts[5] + parts[6] + parts[7])
            print(f"reply_to: {reply_to}, reply_to_timestamp: {reply_to_timestamp}")
            return sender, reply_to, sender_timestamp, reply_to_timestamp
        else:
            return None, None, None, None
    
    if len(parts) >= 4 and len(parts) <= 5:
        
        sender = clean_unicode_formatting(parts[0].replace("_", " ")) if len(parts) == 4 else clean_unicode_formatting(parts[0].replace("_", " ")) + " " + clean_unicode_formatting(parts[1].replace("_", " "))
        reply_to = clean_unicode_formatting(parts[2].replace("_", " ")) if len(parts) == 4 else clean_unicode_formatting(parts[3].replace("_", " ")) 
        sender_timestamp = parts[1] if len(parts) == 4 else parts[2]
        reply_to_timestamp = parts[3] if len(parts) == 4 else parts[4]
        
        if len(parts) > 4 and parts[1].replace("_", " ").isdigit():
            sender = clean_unicode_formatting(parts[0].replace("_", " "))
            reply_to = clean_unicode_formatting(parts[2].replace("_", " ")) + " " + clean_unicode_formatting(parts[3].replace("_", " "))
            sender_timestamp = parts[1]
            reply_to_timestamp = parts[3]
            
        if not reply_to_timestamp.isdigit() and len(parts) > 4:
            reply_to_timestamp = parts[4]

        return sender, reply_to, sender_timestamp, reply_to_timestamp
    elif len(parts) >= 2 and len(parts) <= 3:
        sender = clean_unicode_formatting(parts[0].replace("_", " "))
        timestamp = parts[1]
        return sender, None, timestamp, None
    
    return None, None, None, None

def get_clean_timestamp(timestamp_str):
    try:
        if not timestamp_str:
            return None
        
        if timestamp_str.isdigit():
            return timestamp_str
                    
        timestamp_str = timestamp_str.replace("Z", "")
        timestamp_str = timestamp_str.replace("T", "")
        timestamp_str = timestamp_str.replace(":", "")
        timestamp_str = timestamp_str.replace(" ", "")
        
        if "." in timestamp_str:
            timestamp_str = timestamp_str.split(".")[0]
        return timestamp_str
    except Exception as e:
        logger.error(f"Error in get_clean_timestamp: {timestamp_str}, Exception: {str(e)}")
        return None


def get_direct_element(element):

    try:
        if element is None:
            logger.error("get_direct_element received None element")
            return None
            
        import copy
        temp_element = copy.copy(element)
        
        if temp_element is None:
            logger.error("temp_element is None after copy")
            return None
        
        dl_elements = temp_element.find_all("dl")
        for dl in dl_elements:
            dl.decompose()
        
        return temp_element
        
    except Exception as e:
        logger.error(f"Error in get_direct_element:")
        logger.error(f"Exception: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"element: {element}")
        logger.error(f"element type: {type(element)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def save_wikipedia_data(url):

    try:
        data = process_wiki_talk_page(url)
        for section in data:
            title_counts = {}
            for other_section in data:
                if other_section["title"] in title_counts:
                    title_counts[other_section["title"]] += 1
                else:
                    title_counts[other_section["title"]] = 1
            
            if title_counts[section["title"]] > 1:
                current_count = 0
                for prev_section in data:
                    if prev_section["title"] == section["title"]:
                        current_count += 1
                        if prev_section is section:
                            break
                
                if current_count > 1:
                    section["title"] = f"{section['title']} ({current_count})"
            for i, comment in enumerate(section["comments"]):
                comment["writer_name"] = names_writers[comment["writer_name"]] if comment["writer_name"] in names_writers else comment["writer_name"]
                comment["reply_to"] = names_writers[comment["reply_to"]] if comment["reply_to"] in names_writers else comment["reply_to"]
                if i == 0 and section["title"] and "-" in section["title"]:
                    comment["reply_to"] = None
                if "הודעה זו נכתבה באמצעות מערכת המשוב" in comment["text"]:
                    comment["text"] = comment["text"].replace("הודעה זו נכתבה באמצעות מערכת המשוב", "")
                if "(IDT)תגובה" in comment["text"]:
                    comment["text"] = extract_first_clean_text(comment["text"])
                    
        if data is None:
            logger.error("Failed to extract data - process_wiki_talk_page returned None")
            return None

        output_file = "uploads/wikipedia_data.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Data successfully saved to {output_file}, writers: {names_writers}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error in save_wikipedia_data:")
        logger.error(f"Exception: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

def parse_comment_timestamp(timestamp_str: str) -> datetime:
  
    try:
        if len(timestamp_str) >= 14:
            year = int(timestamp_str[0:4])
            month = int(timestamp_str[4:6])
            day = int(timestamp_str[6:8])
            hour = int(timestamp_str[8:10])
            minute = int(timestamp_str[10:12])
            second = int(timestamp_str[12:14])
            return datetime(year, month, day, hour, minute, second)
        elif len(timestamp_str) >= 8:
            year = int(timestamp_str[0:4])
            month = int(timestamp_str[4:6])
            day = int(timestamp_str[6:8])
            return datetime(year, month, day)
    except (ValueError, IndexError):
        return None
    return None

def create_datetime_range(start_date: str, end_date: str, start_time: Optional[str] = None, end_time: Optional[str] = None):

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if start_time:
            start_time_obj = datetime.strptime(start_time, "%H:%M:%S").time()
            start_dt = datetime.combine(start_dt.date(), start_time_obj)
        
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if end_time:
            end_time_obj = datetime.strptime(end_time, "%H:%M:%S").time()
            end_dt = datetime.combine(end_dt.date(), end_time_obj)
        else:
            end_dt = datetime.combine(end_dt.date(), time(23, 59, 59))
        
        return start_dt, end_dt
    except ValueError as e:
        logger.error(f"Error parsing dates: {e}")
        return None, None

def filter_comments_by_date(comments: list, start_date: str, end_date: str, start_time: Optional[str] = None, end_time: Optional[str] = None):

    if not start_date or not end_date:
        return comments
    
    start_dt, end_dt = create_datetime_range(start_date, end_date, start_time, end_time)
    if not start_dt or not end_dt:
        logger.error("Could not parse date range")
        return comments
    
    filtered_comments = []
    for comment in comments:
        timestamp_str = comment.get("timestamp", "")
        if not timestamp_str:
            continue
            
        comment_dt = parse_comment_timestamp(timestamp_str)
        if comment_dt and start_dt <= comment_dt <= end_dt:
            filtered_comments.append(comment)
    
    logger.info(f"Date filter: {len(comments)} -> {len(filtered_comments)} comments")
    logger.info(f"Date range: {start_dt} to {end_dt}")
    
    return filtered_comments

def extract_first_clean_text(text):
    
    parts = text.split("(IDT)תגובה")
    if not parts:
        return ""
        
    first_part = parts[0]
    
    pattern = r"(.+?)-שיחה(\d{2}:\d{2}, \d{1,2} ב[א-ת]+ \d{4} \(IDT\))$"
    
    match = re.search(pattern, first_part)
    if match:
        return first_part[:match.start()].strip()
    
    return first_part.strip()