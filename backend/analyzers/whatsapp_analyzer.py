import os
import logging
import networkx as nx
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from graph_builder import build_graph_from_txt
from analyzers.base_analyzer import BaseAnalyzer


UPLOAD_FOLDER = "uploads"
logger = logging.getLogger(__name__)


class WhatsAppAnalyzer(BaseAnalyzer):
    async def analyze(self, filename: str, **kwargs):
        try:
            logger.info(f"[WhatsApp] Analyzing (via graph_builder) file: {filename}")
            txt_path = os.path.join(UPLOAD_FOLDER, filename)
            if not os.path.exists(txt_path):
                raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

            kwargs.pop("platform", None)

            graph_data = build_graph_from_txt(txt_path, platform="whatsapp", **kwargs)

            return JSONResponse({
                "nodes": graph_data["nodes"],
                "links": graph_data["links"],
                "messages": graph_data.get("messages") if kwargs.get("is_for_save") else None,
                "is_connected": graph_data.get("is_connected", False)
            })

        except Exception as e:
            logger.error(f"[WhatsApp] analyze_network error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def detect_communities(self, filename: str, **kwargs):
            try:
                from community import community_louvain
                from networkx.algorithms import community as nx_community

                logger.info(f"[WhatsApp] Detecting communities in: {filename}")
                kwargs.pop("platform", None)  

                txt_path = os.path.join(UPLOAD_FOLDER, filename)
                if not os.path.exists(txt_path):
                    raise HTTPException(status_code=404, detail=f"File {txt_path} not found")

                graph_data = build_graph_from_txt(txt_path, platform="whatsapp", **{k: v for k, v in kwargs.items() if k != "algorithm"})

                if not graph_data["nodes"] or not graph_data["links"]:
                    return JSONResponse({
                        "nodes": [],
                        "links": [],
                        "communities": [],
                        "node_communities": {},
                        "algorithm": kwargs.get("algorithm", "louvain"),
                        "num_communities": 0,
                        "modularity": None,
                        "warning": "No data found for community analysis"
                    })

                G = nx.Graph()
                for node in graph_data["nodes"]:
                    G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
                for link in graph_data["links"]:
                    G.add_edge(link["source"], link["target"], weight=link.get("weight", 1))

                algorithm = kwargs.get("algorithm", "louvain")
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

                return JSONResponse({
                    "nodes": graph_data["nodes"],
                    "links": graph_data["links"],
                    "communities": communities_list,
                    "node_communities": node_communities,
                    "algorithm": algorithm,
                    "num_communities": len(communities),
                    "modularity": round(modularity, 4) if modularity else None,
                    "is_connected": graph_data.get("is_connected", False)
                })

            except Exception as e:
                logger.error(f"[WhatsApp] Community detection error: {e}")
                raise HTTPException(status_code=500, detail=str(e))