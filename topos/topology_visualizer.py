import math

import networkx as nx
import matplotlib.pyplot as plt


def visualize_SimAI(path: str):
    """
    Parse a SimAI topology file and draw it.

    File format recap:
      line 1: "<nodes> <gpu_per_server> <nv_switch_num> <(all_switches-nv)> <links> <gpu_type>"
      line 2: "IDs of all switches"  (space-separated)
      line 3+: "<src> <dst> <bandwidth> <latency> <error>"

    Returns:
      G               : networkx.Graph
      ser2sw_links    : list[(u, v)]
      sw2sw_links     : list[(u, v)]
      link_data       : dict[(u, v)] -> {bandwidth, latency, error}
    """
    # ---- read & clean lines ----
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) < 3:
        raise ValueError("File too short: need header, switch list, and at least one link line.")

    # ---- parse header & switch list ----
    hdr = lines[0].split()
    if len(hdr) != 6:
        raise ValueError(f"Header must have 6 fields, got: {lines[0]}")

    try:
        n_nodes = int(hdr[0])  # total nodes (hosts + all switches)
        gpu_per_server = int(hdr[1])
        nv_switch_num = int(hdr[2])
        non_nv_switch_num = int(hdr[3])  # number of switches excluding NV
        n_links_declared = int(hdr[4])
        gpu_type = " ".join(hdr[5:])  # keep tail in case it has spaces
    except Exception as e:
        raise ValueError(f"Cannot parse header: {lines[0]}") from e

    switch_list = [int(x) for x in lines[1].split()]
    nv_switch_ids = switch_list[:nv_switch_num]
    non_nv_switch_set = set(switch_list[nv_switch_num:])
    switch_total = nv_switch_num + non_nv_switch_num
    switch_set = set(switch_list)
    host_ids = [i for i in range(n_nodes) if i not in switch_set]
    assert len(switch_list) == switch_total

    # ---- build graph ----
    G = nx.Graph()
    G.add_nodes_from(host_ids, role="host")
    G.add_nodes_from(switch_list, role="switch")

    ser2sw_links = []
    sw2sw_links = []
    link_data = {}

    # ---- parse edges ----
    asw_set = set()
    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) != 5:
            # skip malformed line
            raise ValueError(f"each line should consist of 5 parts")
        try:
            u, v = map(int, parts[:2])
        except Exception:
            # skip non-edge lines if any slipped through
            continue

        bw = parts[2]
        lat = parts[3]

        link_data[(u, v)] = {"bandwidth": bw, "latency": lat}
        G.add_edge(u, v, **link_data[(u, v)])

        if u in switch_set and v in switch_set:
            sw2sw_links.append((u, v))
        elif u not in switch_set:
            ser2sw_links.append((u, v))
            if v not in nv_switch_ids:
                asw_set.add(v)

    psw_set = non_nv_switch_set - asw_set

    # ---- layout (hosts on one row, switches on another) ----
    psw_list = sorted(psw_set)  # top
    asw_list = sorted(asw_set)
    srv_list = sorted(host_ids)
    nvs_list = sorted(nv_switch_ids)  # bottom

    tiers = [psw_list, asw_list, srv_list, nvs_list]
    y_levels = [3, 2, 1, 0]
    max_width = max(len(t) for t in tiers) or 1

    def x_centered_index(i, n, width=max_width):
        return (width - n) / 2 + i

    pos = {}
    for y, nodes in zip(y_levels, tiers):
        for i, node in enumerate(nodes):
            pos[node] = (x_centered_index(i, len(nodes)), y)

    # ---- draw (order controls layering; no zorder arg needed) ----
    plt.figure(figsize=(14, 7))

    # 1) edges first
    nx.draw_networkx_edges(G, pos, edgelist=ser2sw_links)
    nx.draw_networkx_edges(G, pos, edgelist=sw2sw_links, style='dashed')

    # 2) nodes from bottom to top to make upper tiers sit on top
    nx.draw_networkx_nodes(G, pos, nodelist=nvs_list, node_shape='h', node_size=450, label='NVSwitch')
    nx.draw_networkx_nodes(G, pos, nodelist=srv_list, node_shape='o', node_size=250, label='NIC')
    nx.draw_networkx_nodes(G, pos, nodelist=asw_list, node_shape='s', node_size=350, label='ASW')
    nx.draw_networkx_nodes(G, pos, nodelist=psw_list, node_shape='^', node_size=400, label='PSW')

    # 3) node labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    # 4) edge bandwidth labels
    edge_labels = {}
    for (u, v), meta in link_data.items():
        bw = meta.get("bandwidth", "")
        edge_labels[(u, v)] = bw
        edge_labels[(v, u)] = bw
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5, rotate=False)

    plt.legend(loc="upper right")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return G, ser2sw_links, sw2sw_links, link_data


def visualize_FatTree(path: str, asw_num, psw_num, csw_num):
    """
    Parse a FatTree topology file and draw it.

    File format recap:
      line 1: "<nodes> <gpu_per_server> <nv_switch_num> <(all_switches-nv)> <links> <gpu_type>"
      line 2: "IDs of all switches"  (space-separated)
      line 3+: "<src> <dst> <bandwidth> <latency> <error>"

    Returns:
      G               : networkx.Graph
      ser2sw_links    : list[(u, v)]
      sw2sw_links     : list[(u, v)]
      link_data       : dict[(u, v)] -> {bandwidth, latency, error}
    """
    # ---- read & clean lines ----
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) < 3:
        raise ValueError("File too short: need header, switch list, and at least one link line.")

    # ---- parse header & switch list ----
    hdr = lines[0].split()
    if len(hdr) != 6:
        raise ValueError(f"Header must have 6 fields, got: {lines[0]}")

    try:
        n_nodes = int(hdr[0])  # total nodes (hosts + all switches)
        gpu_per_server = int(hdr[1])
        nv_switch_num = int(hdr[2])
        n_links_declared = int(hdr[4])
        gpu_type = " ".join(hdr[5:])  # keep tail in case it has spaces
    except Exception as e:
        raise ValueError(f"Cannot parse header: {lines[0]}") from e

    switch_list = [int(x) for x in lines[1].split()]
    nv_switch_ids = switch_list[:nv_switch_num]
    asw_set = set(switch_list[nv_switch_num:nv_switch_num + asw_num])
    psw_set = set(switch_list[nv_switch_num + asw_num:nv_switch_num + asw_num + psw_num])
    csw_set = set(switch_list[nv_switch_num + asw_num + psw_num:])
    switch_total = nv_switch_num + asw_num + psw_num + csw_num
    switch_set = set(switch_list)
    host_ids = [i for i in range(n_nodes) if i not in switch_set]
    assert len(switch_list) == switch_total

    # ---- build graph ----
    G = nx.Graph()
    G.add_nodes_from(host_ids, role="host")
    G.add_nodes_from(switch_list, role="switch")

    ser2sw_links = []
    sw2sw_links = []
    link_data = {}

    # ---- parse edges ----
    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) != 5:
            # skip malformed line
            raise ValueError(f"each line should consist of 5 parts")
        try:
            u, v = map(int, parts[:2])
        except Exception:
            # skip non-edge lines if any slipped through
            continue

        bw = parts[2]
        lat = parts[3]

        link_data[(u, v)] = {"bandwidth": bw, "latency": lat}
        G.add_edge(u, v, **link_data[(u, v)])

        if u in switch_set and v in switch_set:
            sw2sw_links.append((u, v))
        elif u not in switch_set:
            ser2sw_links.append((u, v))

    # ---- layout (hosts on one row, switches on another) ----
    csw_list = sorted(csw_set)  # top
    psw_list = sorted(psw_set)
    asw_list = sorted(asw_set)
    srv_list = sorted(host_ids)
    nvs_list = sorted(nv_switch_ids)  # bottom

    tiers = [csw_list, psw_list, asw_list, srv_list, nvs_list]
    y_levels = [4, 3, 2, 1, 0]
    max_width = max(len(t) for t in tiers) or 1

    def x_centered_index(i, n, width=max_width):
        return (width - n) / 2 + i

    pos = {}
    for y, nodes in zip(y_levels, tiers):
        for i, node in enumerate(nodes):
            pos[node] = (x_centered_index(i, len(nodes)), y)

    # ---- draw (order controls layering; no zorder arg needed) ----
    plt.figure(figsize=(14, 7))

    # 1) edges first
    nx.draw_networkx_edges(G, pos, edgelist=ser2sw_links)
    nx.draw_networkx_edges(G, pos, edgelist=sw2sw_links, style='dashed')

    # 2) nodes from bottom to top to make upper tiers sit on top
    nx.draw_networkx_nodes(G, pos, nodelist=nvs_list, node_shape='h', node_size=450, label='NVSwitch')
    nx.draw_networkx_nodes(G, pos, nodelist=srv_list, node_shape='o', node_size=250, label='NIC')
    nx.draw_networkx_nodes(G, pos, nodelist=asw_list, node_shape='s', node_size=350, label='LEAF')
    nx.draw_networkx_nodes(G, pos, nodelist=psw_list, node_shape='^', node_size=400, label='AGG')
    nx.draw_networkx_nodes(G, pos, nodelist=csw_list, node_shape='*', node_size=400, label='CORE')

    # 3) node labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    # 4) edge bandwidth labels
    edge_labels = {}
    for (u, v), meta in link_data.items():
        bw = meta.get("bandwidth", "")
        edge_labels[(u, v)] = bw
        edge_labels[(v, u)] = bw
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5, rotate=False)

    plt.legend(loc="upper right")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return G, ser2sw_links, sw2sw_links, link_data


def visualize_UBMesh(path: str):
    """
    Parse a UB-Mesh topology file (64 GPUs, no switches) and draw it.
    Layout:
      - 8 boards arranged in a big octagon.
      - Each board: 8 GPUs arranged in a small octagon around the board center.
      - Intra-board edges: solid.
      - Inter-board same-index edges: dashed.
    """
    # ---- read lines ----
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    hdr = lines[0].split()
    n_nodes = int(hdr[0])

    # ---- build graph ----
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes), role="host")

    link_data = {}
    intra_edges = []
    inter_edges = []

    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) != 5:
            continue
        u, v = map(int, parts[:2])
        bw, lat = parts[2], parts[3]
        link_data[(u, v)] = {"bandwidth": bw, "latency": lat}
        G.add_edge(u, v, **link_data[(u, v)])
        if u // 8 == v // 8:
            intra_edges.append((u, v))
        else:
            inter_edges.append((u, v))

    # ---- layout ----
    pos = {}
    big_radius = 15.0  # rack
    small_radius = 3.0  # board

    # board center positions
    board_centers = {}
    for b in range(8):
        angle_b = 2 * math.pi * b / 8
        cx = big_radius * math.cos(angle_b)
        cy = big_radius * math.sin(angle_b)
        board_centers[b] = (cx, cy)

        for i in range(8):
            angle_g = 2 * math.pi * i / 8
            x = cx + small_radius * math.cos(angle_g)
            y = cy + small_radius * math.sin(angle_g)
            node_id = b * 8 + i
            pos[node_id] = (x, y)

    # ---- draw ----
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(G, pos, edgelist=intra_edges, edge_color="tab:blue", width=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=inter_edges, style="dashed", alpha=0.4)

    nx.draw_networkx_nodes(G, pos, nodelist=range(n_nodes), node_shape='o', node_size=250, label='GPU')
    nx.draw_networkx_labels(G, pos, font_size=7)

    plt.title("UB-Mesh 2D Full-Mesh (64 GPUs)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


visualize_UBMesh("./UB-Mesh_rack_64g")
# visualize_SimAI("./Rail_Opti_SingleToR_32g_8gps_400Gbps_H100")
# visualize_FatTree('FatTree_16g_2gps_200Gbps_H100', 8, 8, 4)
