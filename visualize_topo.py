import os
import math
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_ring_allreduce_info(gpu_per_server):
    """
    Ring AllReduce Paths:
    Job 1 (Blue): 3 -> 2 -> 1 -> 5 -> 4 -> 0 -> 3
    Job 2 (Pink): 8 -> 9 -> 10 -> 6 -> 7 -> 11 -> 8
    """
    job1_gpus = {0, 1, 2, 3, 4, 5}
    job1_logical = [(3, 2), (2, 1), (1, 5), (5, 4), (4, 0), (0, 3)]
    
    job2_gpus = {6, 7, 8, 9, 10, 11}
    job2_logical = [(8, 9), (9, 10), (10, 6), (6, 7), (7, 11), (11, 8)]
    
    job1_links = set()
    job2_links = set()
    
    def resolve_physical_links(logical_edges, target_links_set):
        for u, v in logical_edges:
            u_server = u // gpu_per_server
            v_server = v // gpu_per_server
            if u_server == v_server:
                nv_switch = 16 + u_server
                target_links_set.add((min(u, nv_switch), max(u, nv_switch)))
                target_links_set.add((min(v, nv_switch), max(v, nv_switch)))
            else:
                rail_idx = u % gpu_per_server
                asw_id = 20 + rail_idx
                target_links_set.add((min(u, asw_id), max(u, asw_id)))
                target_links_set.add((min(v, asw_id), max(v, asw_id)))

    resolve_physical_links(job1_logical, job1_links)
    resolve_physical_links(job2_logical, job2_links)
    
    return job1_gpus, job1_links, job2_gpus, job2_links

def get_ep_cross_rail_info():
    """
    Scenario 2: EP Cross-rail traffic under a Good ECMP hashing (Disjoint Spines)
    Job 1 (Blue): GPU 0 (Rail 0) -> ASW 20 -> PSW 24 -> ASW 21 -> GPU 5 (Rail 1)
    Job 2 (Pink): GPU 8 (Rail 0) -> ASW 20 -> PSW 25 -> ASW 23 -> GPU 7 (Rail 3)
    No links are shared because they route through different Spine switches (PSW 24 vs 25).
    """
    job1_gpus = {0, 5}
    job1_links = {(0, 20), (20, 24), (21, 24), (5, 21)}
    
    job2_gpus = {8, 7}
    job2_links = {(8, 20), (20, 25), (23, 25), (7, 23)}
    
    return job1_gpus, job1_links, job2_gpus, job2_links

def get_spine_collision_info():
    """
    Scenario 3: EP Cross-rail traffic under Bad ECMP hashing (ECMP Hash Collision)
    Job 1 (Blue): GPU 0 (Rail 0) -> ASW 20 -> PSW 24 -> ASW 21 -> GPU 5 (Rail 1)
    Job 2 (Pink): GPU 8 (Rail 0) -> ASW 20 -> PSW 24 -> ASW 23 -> GPU 7 (Rail 3)
    Shared Link (Red): ASW 20 -> PSW 24 (Uplink bandwidth competition)
    """
    job1_gpus = {0, 5}
    job1_links = {(0, 20), (21, 24), (5, 21)}  # Shared (20, 24) excluded
    
    job2_gpus = {8, 7}
    job2_links = {(8, 20), (23, 24), (7, 23)}  # Shared (20, 24) excluded
    
    shared_links = {(20, 24)}
    return job1_gpus, job1_links, job2_gpus, job2_links, shared_links

def draw_topology(file_path, output_image_path, mode='showcase'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Topology file not found: {file_path}")

    # Read and clean lines
    with open(file_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) < 3:
        raise ValueError("File too short.")

    # Parse header
    hdr = lines[0].split()
    n_nodes = int(hdr[0])
    gpu_per_server = int(hdr[1])
    nv_switch_num = int(hdr[2])
    non_nv_switch_num = int(hdr[3])
    gpu_type = hdr[5]

    # Parse switches list
    switch_list = [int(x) for x in lines[1].split()]
    nv_switch_ids = switch_list[:nv_switch_num]
    non_nv_switch_set = set(switch_list[nv_switch_num:])
    switch_set = set(switch_list)
    host_ids = [i for i in range(n_nodes) if i not in switch_set]
    num_servers = len(host_ids) // gpu_per_server

    # Create networkx Graph
    G = nx.Graph()
    G.add_nodes_from(host_ids, role="host")
    G.add_nodes_from(switch_list, role="switch")

    # Links categorization
    nv_links = []
    nic_links = {r: [] for r in range(gpu_per_server)}
    spine_leaf_links = []

    # Parse links
    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) < 2:
            continue
        u, v = int(parts[0]), int(parts[1])
        
        # Link classification
        if u in switch_set and v in switch_set:
            spine_leaf_links.append((u, v))
        elif u in nv_switch_ids or v in nv_switch_ids:
            nv_links.append((u, v))
        else:
            gpu_id = u if u not in switch_set else v
            rail_idx = gpu_id % gpu_per_server
            nic_links[rail_idx].append((u, v))
            G.add_edge(u, v)

    # Add NV and Spine-Leaf links to graph
    for u, v in nv_links:
        G.add_edge(u, v)
    for u, v in spine_leaf_links:
        G.add_edge(u, v)

    # Calculate coordinates dynamically
    pos = {}
    for server_idx in range(num_servers):
        server_x_offset = server_idx * (gpu_per_server + 1.5)
        for gpu_in_server in range(gpu_per_server):
            gpu_id = server_idx * gpu_per_server + gpu_in_server
            pos[gpu_id] = (server_x_offset + gpu_in_server, 1.0)

    nv_switches_per_server = nv_switch_num // num_servers if num_servers > 0 else 0
    for s in range(num_servers):
        server_x_offset = s * (gpu_per_server + 1.5)
        server_center_x = server_x_offset + (gpu_per_server - 1) / 2.0
        for nv_idx in range(nv_switches_per_server):
            nv_id = nv_switch_ids[s * nv_switches_per_server + nv_idx]
            pos[nv_id] = (server_center_x, 0.0)

    asw_set = set()
    for r in range(gpu_per_server):
        for u, v in nic_links[r]:
            asw_id = u if u in switch_set else v
            asw_set.add(asw_id)
    asw_list = sorted(list(asw_set))
    psw_list = sorted(list(non_nv_switch_set - asw_set))

    total_width = num_servers * (gpu_per_server + 1.5) - 1.5
    for idx, asw_id in enumerate(asw_list):
        x_pos = idx * (gpu_per_server + 1.5) + (gpu_per_server - 1) / 2.0
        pos[asw_id] = (x_pos, 2.2)

    psw_count = len(psw_list)
    for idx, psw_id in enumerate(psw_list):
        if psw_count > 1:
            x_pos = (idx / (psw_count - 1)) * (total_width * 0.6) + (total_width * 0.2)
        else:
            x_pos = total_width / 2.0
        pos[psw_id] = (x_pos, 3.4)

    # Style templates
    rail_colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f97316', '#10b981', '#f59e0b', '#ef4444', '#14b8a6']
    
    asw_rail_map = {}
    for r in range(gpu_per_server):
        for u, v in nic_links[r]:
            asw_id = u if u in switch_set else v
            asw_rail_map[asw_id] = r

    def get_showcase_link_style(u, v):
        if (u, v) in nv_links or (v, u) in nv_links:
            return "#06b6d4", 3.2, "solid"  # NVLink
        elif (u, v) in spine_leaf_links or (v, u) in spine_leaf_links:
            return "#475569", 1.0, "dashed"  # Spine-Leaf
        else:
            gpu_id = u if u not in switch_set else v
            rail = gpu_id % gpu_per_server
            return rail_colors[rail % len(rail_colors)], 2.0, "solid"

    # Resolve jobs info based on mode
    shared_links = set()
    if mode == 'highlight':
        job1_gpus, job1_links, job2_gpus, job2_links = get_ring_allreduce_info(gpu_per_server)
        active_nv = {16, 17, 18}
        active_asw = {20, 21, 22, 23}
        active_psw = set()
    elif mode == 'ep_cross_rail':
        job1_gpus, job1_links, job2_gpus, job2_links = get_ep_cross_rail_info()
        active_nv = set()
        active_asw = {20, 21, 23}
        active_psw = {24, 25}
    elif mode == 'spine_collision':
        job1_gpus, job1_links, job2_gpus, job2_links, shared_links = get_spine_collision_info()
        active_nv = set()
        active_asw = {20, 21, 23}
        active_psw = {24}
    else:
        job1_gpus, job1_links, job2_gpus, job2_links = set(), set(), set(), set()
        active_nv, active_asw, active_psw = set(), set(), set()

    # Set up matplotlib figure
    bg_color = "#0b0f19"
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.axis("off")

    # Draw physical server boxes (chassis)
    for s in range(num_servers):
        has_highlight = False
        if mode != 'showcase':
            for gpu_in_server in range(gpu_per_server):
                gpu_id = s * gpu_per_server + gpu_in_server
                if gpu_id in job1_gpus or gpu_id in job2_gpus:
                    has_highlight = True
                    break
        
        box_alpha = 0.6 if (mode == 'showcase' or has_highlight) else 0.15
        border_style = "--" if (mode == 'showcase' or has_highlight) else ":"
        border_color = "#334155" if (mode == 'showcase' or has_highlight) else "#1e293b"
        text_color = "#94a3b8" if (mode == 'showcase' or has_highlight) else "#475569"

        server_x_offset = s * (gpu_per_server + 1.5)
        left = server_x_offset - 0.4
        width = gpu_per_server - 0.2
        bottom = -0.4
        height = 1.8
        
        rect = patches.Rectangle(
            (left, bottom), width, height,
            facecolor="#1e293b",
            edgecolor=border_color,
            linewidth=2.0,
            linestyle=border_style,
            alpha=box_alpha,
            zorder=0
        )
        ax.add_patch(rect)
        
        ax.text(
            left + width/2.0, bottom + 0.12,
            f"Server {s}",
            color=text_color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=1
        )

    # --- Draw Links ---
    if mode == 'showcase':
        for u, v in G.edges():
            col, width, style = get_showcase_link_style(u, v)
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                edge_color=col, width=width, style=style, alpha=0.6, ax=ax
            )
    else:  # highlight or ep_cross_rail or spine_collision
        # Multi-pass drawing for layering: dimmed first, active second, shared last
        dimmed_list = []
        job1_list = []
        job2_list = []
        shared_list = []
        
        for u, v in G.edges():
            edge_key = (min(u, v), max(u, v))
            showcase_col, width, style = get_showcase_link_style(u, v)
            
            if edge_key in shared_links:
                shared_list.append((u, v, width))
            elif edge_key in job1_links:
                job1_list.append((u, v, width))
            elif edge_key in job2_links:
                job2_list.append((u, v, width))
            else:
                dimmed_list.append((u, v, showcase_col, width, style))
                
        # 1. Dimmed links
        for u, v, col, w, s in dimmed_list:
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                edge_color=col, width=w, style=s, alpha=0.12, ax=ax
            )
        # 2. Job 1 links (Blue)
        for u, v, w in job1_list:
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                edge_color="#3b82f6", width=w, style="solid", alpha=1.0, ax=ax
            )
        # 3. Job 2 links (Pink)
        for u, v, w in job2_list:
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                edge_color="#ec4899", width=w, style="solid", alpha=1.0, ax=ax
            )
        # 4. Shared links (Red)
        for u, v, w in shared_list:
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                edge_color="#ef4444", width=w, style="solid", alpha=1.0, ax=ax
            )

    # --- Draw Nodes ---
    def draw_nodes_by_style(nodelist, color, shape, size, alpha=1.0, edgecolor="#ffffff", lw=1.0):
        if nodelist:
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodelist,
                node_color=color, node_shape=shape, node_size=size,
                edgecolors=edgecolor, linewidths=lw, alpha=alpha, ax=ax
            )

    if mode == 'showcase':
        # GPUs
        for gpu_id in host_ids:
            draw_nodes_by_style([gpu_id], rail_colors[gpu_id % gpu_per_server], "o", 400)
        # NV Switches
        draw_nodes_by_style(nv_switch_ids, "#0891b2", "d", 350)
        # ASWs
        for asw_id in asw_list:
            draw_nodes_by_style([asw_id], rail_colors[asw_rail_map[asw_id]], "s", 450)
        # PSWs
        draw_nodes_by_style(psw_list, "#475569", "^", 500)
    else:  # highlight, ep_cross_rail, spine_collision
        # 1. Job 1 GPUs
        draw_nodes_by_style(list(job1_gpus), "#3b82f6", "o", 400, alpha=1.0, lw=1.5)
        # 2. Job 2 GPUs
        draw_nodes_by_style(list(job2_gpus), "#ec4899", "o", 400, alpha=1.0, lw=1.5)
        
        # 3. Unused GPUs
        unused_gpus = [g for g in host_ids if g not in job1_gpus and g not in job2_gpus]
        for g in unused_gpus:
            draw_nodes_by_style([g], rail_colors[g % gpu_per_server], "o", 400, alpha=0.2, edgecolor="#334155")
            
        # 4. NV Switches
        high_nvs = [nv for nv in nv_switch_ids if nv in active_nv]
        dim_nvs = [nv for nv in nv_switch_ids if nv not in active_nv]
        draw_nodes_by_style(high_nvs, "#0891b2", "d", 350, alpha=1.0)
        draw_nodes_by_style(dim_nvs, "#0891b2", "d", 350, alpha=0.2, edgecolor="#334155")
        
        # 5. ASW Switches
        high_asws = [asw for asw in asw_list if asw in active_asw]
        dim_asws = [asw for asw in asw_list if asw not in active_asw]
        for asw in high_asws:
            draw_nodes_by_style([asw], rail_colors[asw_rail_map[asw]], "s", 450, alpha=1.0)
        for asw in dim_asws:
            draw_nodes_by_style([asw], rail_colors[asw_rail_map[asw]], "s", 450, alpha=0.2, edgecolor="#334155")
            
        # 6. PSW Switches
        high_psws = [psw for psw in psw_list if psw in active_psw]
        dim_psws = [psw for psw in psw_list if psw not in active_psw]
        draw_nodes_by_style(high_psws, "#475569", "^", 500, alpha=1.0)
        draw_nodes_by_style(dim_psws, "#475569", "^", 500, alpha=0.2, edgecolor="#334155")

    # --- Draw Labels ---
    labels = {node: str(node) for node in G.nodes()}
    if mode == 'showcase':
        nx.draw_networkx_labels(
            G, pos, labels=labels, font_size=8,
            font_color="#ffffff", font_weight="bold", ax=ax
        )
    else:
        active_nodes = job1_gpus.union(job2_gpus).union(active_nv).union(active_asw).union(active_psw)
        dim_labels = {n: str(n) for n in G.nodes() if n not in active_nodes}
        nx.draw_networkx_labels(
            G, pos, labels=dim_labels, font_size=8,
            font_color="#64748b", font_weight="normal", alpha=0.4, ax=ax
        )
        high_labels = {n: str(n) for n in G.nodes() if n in active_nodes}
        nx.draw_networkx_labels(
            G, pos, labels=high_labels, font_size=8,
            font_color="#ffffff", font_weight="bold", ax=ax
        )

    # --- Create Legend ---
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e2e8f0', markersize=10, label='GPU (NIC)'),
        plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='#0891b2', markersize=10, label='NV Switch'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#cbd5e1', markersize=10, label='ASW (Leaf)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#475569', markersize=10, label='PSW (Spine)'),
    ]
    
    if mode == 'showcase':
        legend_elements.extend([
            plt.Line2D([0], [0], color='#06b6d4', lw=3, label='NVLink (2880Gbps)'),
            plt.Line2D([0], [0], color='#475569', lw=1, linestyle='--', label='Spine-Leaf (400Gbps)'),
        ])
        for r in range(gpu_per_server):
            legend_elements.append(
                plt.Line2D([0], [0], color=rail_colors[r % len(rail_colors)], lw=2, label=f'Rail {r} Link')
            )
    elif mode == 'highlight':
        legend_elements.extend([
            plt.Line2D([0], [0], color='#3b82f6', lw=2.5, label='Job 1 Ring AllReduce Path'),
            plt.Line2D([0], [0], color='#ec4899', lw=2.5, label='Job 2 Ring AllReduce Path'),
            plt.Line2D([0], [0], color='#334155', lw=1.5, linestyle='--', alpha=0.6, label='Dimmed (Unused) Links'),
        ])
    elif mode == 'ep_cross_rail':
        legend_elements.extend([
            plt.Line2D([0], [0], color='#3b82f6', lw=2.5, label='Job 1 EP Cross-Rail Route'),
            plt.Line2D([0], [0], color='#ec4899', lw=2.5, label='Job 2 EP Cross-Rail Route'),
            plt.Line2D([0], [0], color='#334155', lw=1.5, linestyle='--', alpha=0.6, label='Dimmed (Unused) Links'),
        ])
    elif mode == 'spine_collision':
        legend_elements.extend([
            plt.Line2D([0], [0], color='#3b82f6', lw=2.5, label='Job 1 Routing'),
            plt.Line2D([0], [0], color='#ec4899', lw=2.5, label='Job 2 Routing'),
            plt.Line2D([0], [0], color='#ef4444', lw=3.0, label='ECMP Link Collision (Uplink Shared)'),
            plt.Line2D([0], [0], color='#334155', lw=1.5, linestyle='--', alpha=0.6, label='Dimmed (Unused) Links'),
        ])

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#cbd5e1",
        fontsize=9,
        framealpha=0.9
    )

    # --- Title ---
    if mode == 'showcase':
        ax.set_title(
            f"Rail-Optimized Topology: {num_servers} Servers × {gpu_per_server} GPUs ({gpu_type})",
            color="#ffffff",
            fontsize=16,
            fontweight="bold",
            pad=20
        )
    elif mode == 'highlight':
        ax.set_title(
            f"Parallel 6-GPU Ring AllReduce Job Hops in a Rail-Optimized Network",
            color="#ffffff",
            fontsize=16,
            fontweight="bold",
            pad=20
        )
    elif mode == 'ep_cross_rail':
        ax.set_title(
            f"Expert Parallelism (EP) Cross-Rail Traffic Routing",
            color="#ffffff",
            fontsize=16,
            fontweight="bold",
            pad=20
        )
    elif mode == 'spine_collision':
        ax.set_title(
            f"Spine Uplink Bandwidth Competition (ECMP Hash Collision)",
            color="#ffffff",
            fontsize=16,
            fontweight="bold",
            pad=20
        )

    plt.tight_layout()
    
    out_dir = os.path.dirname(output_image_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    plt.savefig(output_image_path, dpi=300, facecolor=bg_color)
    plt.close()
    print(f"Topology saved successfully to {output_image_path}")

if __name__ == "__main__":
    topo_file = "topos/Rail_Opti_SingleToR_16g_4gps_400Gbps_H100"
    
    # 1. Showcase
    draw_topology(topo_file, "topos/Rail_Opti_16g_4gps_topology.png", mode='showcase')
    
    # 2. Ring AllReduce
    draw_topology(topo_file, "topos/Rail_Opti_16g_4gps_highlight.png", mode='highlight')
    
    # 3. EP Cross Rail
    draw_topology(topo_file, "topos/Rail_Opti_16g_4gps_ep_cross_rail.png", mode='ep_cross_rail')
    
    # 4. Spine Collision
    draw_topology(topo_file, "topos/Rail_Opti_16g_4gps_spine_collision.png", mode='spine_collision')
