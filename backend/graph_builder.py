import re
import networkx as nx
from datetime import datetime
from collections import defaultdict
import logging
from utils import (
    calculate_sequential_weights, 
    normalize_links_by_target,
    parse_date_time,
    MEDIA_RE,
    spam_messages
)

logger = logging.getLogger("graph_builder")


def debug_check_target_weights(links):
    target_totals = defaultdict(float)
    for link in links:
        target_totals[link["target"]] += link["weight"]

    for target, total in target_totals.items():
        print(f"Target: {target}, Total Weight: {round(total, 4)}")


def build_graph_from_txt(
    txt_path,
    limit=None,
    limit_type="first",
    min_length=None,
    max_length=None,
    anonymize=False,
    keywords=None,
    min_messages=None,
    max_messages=None,
    active_users=None,
    selected_users=None,
    username=None,
    start_date=None,
    start_time=None,
    end_date=None,
    end_time=None,
    directed=False,
    use_history=False,
    include_messages=False,
    normalize=False,
    history_length=3,
    message_weights=None,
    is_for_save=False,
    platform="whatsapp",
    algorithm=None
):
    print(f"\n=== GRAPH BUILDING DEBUG ===")
    print(f"Platform: {platform}")
    print(f"File: {txt_path}")
    print(f"Raw parameters:")
    print(f"  start_date: '{start_date}' (type: {type(start_date)})")
    print(f"  start_time: '{start_time}' (type: {type(start_time)})")
    print(f"  end_date: '{end_date}' (type: {type(end_date)})")
    print(f"  end_time: '{end_time}' (type: {type(end_time)})")
    print(f"  limit: {limit}")
    print(f"  keywords: {keywords}")
    print(f"  username: {username}")

    def parse_whatsapp_datetime(date_str, time_str):

        if '.' in date_str:
            parts = date_str.split('.')
            if len(parts) == 3:
                day = parts[0].zfill(2)
                month = parts[1].zfill(2)
                year = parts[2]
                date_str = f"{day}.{month}.{year}"
        
        if time_str.count(':') == 1:
            time_str = f"{time_str}:00"
        
        timestamp_str = f"{date_str} {time_str}"
        
        formats_to_try = [
            "%d.%m.%Y %H:%M:%S",    
            "%d.%m.%Y %H:%M",       
            "%d/%m/%Y %H:%M:%S",    
            "%d/%m/%Y %H:%M",       
            "%m/%d/%Y %H:%M:%S",    
            "%m/%d/%Y %H:%M",       
            "%d.%m.%Y, %H:%M:%S",   
            "%d.%m.%Y, %H:%M",
            "%m/%d/%y %H:%M:%S", 
            "%m/%d/%y %H:%M",
        ]
        
        for fmt in formats_to_try:
            try:
                parsed_dt = datetime.strptime(timestamp_str, fmt)
                return parsed_dt
            except ValueError:
                continue
        
        print(f"Failed to parse datetime: '{timestamp_str}'")
        return None

    def process_whatsapp_messages(lines, filters):        
        print(f"Processing {len(lines)} lines as WhatsApp format...")
        
        filtered_lines = []
        processed_count = 0
        date_filter_skipped = 0
        
        start_dt = filters.get('start_dt')
        end_dt = filters.get('end_dt')
        
        for line_num, line in enumerate(lines):
            line = line.replace('\u202f', ' ').replace('\u200f', ' ').replace('\u200e', ' ').strip()
            
            match = None
            
            if not match:
                match = re.match(r"\[(\d{1,2}[./]\d{1,2}[./]\d{4}), (\d{1,2}:\d{2}(?::\d{2})?)\] *[~\s\u200f\u202f\u200e]*([^:]+):(.*)", line)
            
            if not match:
                match = re.match(r"\[(\d{1,2}\.\d{1,2}\.\d{4}), (\d{2}:\d{2})\] ([^:]+):(.*)", line)
            
            if not match:
                match = re.match(r"(\d{1,2}/\d{1,2}/\d{2,4}), (\d{2}:\d{2}) - ([^:]+):(.*)", line)
            
            if not match:
                match = re.match(r"(\d{1,2}/\d{1,2}/\d{2,4}), (\d{2}:\d{2}:\d{2}) - ([^:]+):(.*)", line)
            
            if not match:
                continue
            
            date, time, user, text = match.groups()
            text = text.strip()
            user = user.strip().lstrip('~ ').replace('\u202f', '').replace('\u200f', '').replace('\u200e', '')
            
            processed_count += 1
            
            if line_num < 5:
                print(f"Line {line_num+1}: Date='{date}', Time='{time}', User='{user}', Text='{text[:30]}...'")
            
            min_len = filters.get('min_length')
            max_len = filters.get('max_length')
            if (min_len is not None and min_len != '' and len(text) < int(min_len)) or \
               (max_len is not None and max_len != '' and len(text) > int(max_len)):
                continue
            
            keywords = filters.get('keywords')
            if keywords:
                keyword_list = [k.strip().lower() for k in keywords.split(",")]
                if not any(k in text.lower() for k in keyword_list):
                    continue
            
            message_dt = parse_whatsapp_datetime(date, time)
            if not message_dt:
                if line_num < 5:
                    print(f"Failed to parse datetime: {date} {time}")
                continue
            
            if line_num < 5:
                print(f"Parsed datetime: {message_dt}")
                if start_dt:
                    print(f"  Start check: {message_dt} >= {start_dt} = {message_dt >= start_dt}")
                if end_dt:
                    print(f"  End check: {message_dt} <= {end_dt} = {message_dt <= end_dt}")
            
            if start_dt and message_dt < start_dt:
                date_filter_skipped += 1
                if line_num < 5:
                    print(f"SKIPPED: Before start date")
                continue
                
            if end_dt and message_dt > end_dt:
                date_filter_skipped += 1
                if line_num < 5:
                    print(f"SKIPPED: After end date")
                continue
            
            try:
                if any(spam in text for spam in spam_messages) or MEDIA_RE.search(text):
                    continue
            except NameError:
                pass
            
            username_filter = filters.get('username')
            if username_filter and username_filter.strip().lower() != user.lower():
                continue
            
            if line_num < 5:
                print(f"ACCEPTED")
            
            filtered_lines.append((user, text))
        
        print(f"WhatsApp processing summary:")
        print(f"  Lines processed: {processed_count}")
        print(f"  Skipped by date filter: {date_filter_skipped}")
        print(f"  Final accepted messages: {len(filtered_lines)}")
        
        return filtered_lines

    def process_wikipedia_messages(lines, filters):
        
        print(f"Processing {len(lines)} lines as Wikipedia format...")
        
        # כאן תוכל להוסיף לוגיקה ספציפית לעיבוד ויקיפדיה
        # לעת עתה, נחזיר רשימה ריקה או נטפל בפורמט הבסיסי
        
        filtered_lines = []
        
        for line in lines:
            if line.strip() and not line.startswith('#'):
                filtered_lines.append(("WikiUser", line.strip()))
        
        print(f"Wikipedia processing summary: {len(filtered_lines)} messages")
        return filtered_lines

    usernames = set()
    user_message_count = defaultdict(int)
    edges_counter = defaultdict(int)
    anonymized_map = {}

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Total lines in file: {len(lines)}")

    print(f"Processing date filters:")
    print(f"  start_date: '{start_date}', start_time: '{start_time}'")
    print(f"  end_date: '{end_date}', end_time: '{end_time}'")

    start_dt = None
    if start_date:
        start_dt = parse_date_time(start_date, start_time)

    end_dt = None
    if end_date:
        if start_date == end_date and (not end_time or end_time == "None" or not end_time.strip()):
            print(f"Same day detected! Using full day for: {end_date}")
            end_dt = parse_date_time(end_date, "23:59:59")
        else:
            end_dt = parse_date_time(end_date, end_time)

    print(f"Final parsed date filters:")
    print(f"  start_dt: {start_dt}")
    print(f"  end_dt: {end_dt}")

    if start_date == end_date and start_dt and end_dt:
        print(f"Filtering for entire day: {start_date}")
        print(f"From: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"To:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    filter_params = {
        'start_dt': start_dt,
        'end_dt': end_dt,
        'min_length': min_length,
        'max_length': max_length,
        'keywords': keywords,
        'username': username
    }

    if platform == "whatsapp":
        filtered_lines = process_whatsapp_messages(lines, filter_params)
    elif platform == "wikipedia":
        filtered_lines = process_wikipedia_messages(lines, filter_params)
    else:
        print(f"Unknown platform: {platform}, defaulting to WhatsApp")
        filtered_lines = process_whatsapp_messages(lines, filter_params)

    print(f"Messages after platform-specific filtering: {len(filtered_lines)}")

    if directed and use_history:
        if limit:
            if limit_type == "last":
                filtered_lines = filtered_lines[-limit:]
            else:
                filtered_lines = filtered_lines[:limit]
        else:
            if limit_type == "last":
                filtered_lines = filtered_lines[::-1]
            else:
                filtered_lines = filtered_lines
    else:
        if limit:
            if limit_type == "last":
                filtered_lines = filtered_lines[-limit:][::-1]
            else:
                filtered_lines = filtered_lines[:limit]
        else:
            if limit_type == "last":
                filtered_lines = filtered_lines[::-1]
            else:
                filtered_lines = filtered_lines
        
    all_messages = []
    for user, text in filtered_lines:
        if anonymize:
            if user not in anonymized_map:
                anonymized_map[user] = f"User_{len(anonymized_map) + 1}"
            user = anonymized_map[user]
        
        usernames.add(user)
        user_message_count[user] += 1
        all_messages.append((user, text))

    print(f"Unique users found: {len(usernames)}")
    print(f"Total messages: {len(all_messages)}")

    if directed and use_history:
        print("Using history algorithm for edge weights...")
        history_n = int(history_length) if history_length else 3
        edges = calculate_sequential_weights(all_messages, n_prev=history_n, message_weights=message_weights)
        for (source, target), weight in edges.items():
            edge = (source, target)
            edges_counter[edge] += weight
    else:
        print("Using simple sequential algorithm...")
        previous_user = None
        for user, _ in all_messages:
            if previous_user and previous_user != user:
                edge = (user, previous_user) if directed else tuple(sorted([user, previous_user]))
                edges_counter[edge] += 1
            previous_user = user

    if min_messages or max_messages or active_users or selected_users:
        print("Applying user filters...")
        
        filtered_users = {u: c for u, c in user_message_count.items()
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
        print(f"Users after filtering: {len(usernames)}")

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(usernames)

    for edge, weight in edges_counter.items():
        if edge[0] in usernames and edge[1] in usernames:
            G.add_edge(edge[0], edge[1], weight=weight)

    print(f"Graph built: {len(G.nodes())} nodes, {len(G.edges())} edges")

    if directed:
        is_connected = nx.is_weakly_connected(G) if len(G.nodes()) > 0 else False
    else:
        is_connected = nx.is_connected(G) if len(G.nodes()) > 0 else False

    print(f"Graph is connected: {is_connected}")

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
            "messages": user_message_count.get(user, 0),
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
    
    if normalize:
        print("Normalizing link weights by target...")
        links_list = normalize_links_by_target(links_list)
        debug_check_target_weights(links_list)

    print(f"\n=== FINAL RESULTS ===")
    print(f"Nodes: {len(nodes_list)}")
    print(f"Links: {len(links_list)}")
    print(f"Platform: {platform}")
    print(f"Date range applied: {start_dt} to {end_dt}")

    return {
        "nodes": nodes_list,
        "links": links_list,
        "is_connected": is_connected,
        "messages": all_messages
    }