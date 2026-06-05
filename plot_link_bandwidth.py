import sys
import struct
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_interface_indices(topo_path, src, dst):
    if not os.path.exists(topo_path):
        print(f"Topology file not found: {topo_path}")
        return None, None
    
    src_occurrences = 0
    dst_occurrences = 0
    link_found = False
    
    with open(topo_path, 'r') as f:
        # Skip header
        lines = [line.strip() for line in f if line.strip()]
        if len(lines) < 3:
            return None, None
            
        # Parse links starting from line 2
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            
            # Count occurrences of u and v
            # Since NS-3 devices are added sequentially,
            # loopback is index 0, so the c-th link installed on a node gets index c.
            if u == src or v == src:
                src_occurrences += 1
            if u == dst or v == dst:
                dst_occurrences += 1
                
            if (u == src and v == dst) or (u == dst and v == src):
                link_found = True
                src_port = src_occurrences
                dst_port = dst_occurrences
                
    if link_found:
        return src_port, dst_port
    else:
        return None, None

def analyze_and_plot(trace_path, topo_path, src, dst, bin_size_us=100.0, output_img="link_bandwidth.png"):
    src_port, dst_port = get_interface_indices(topo_path, src, dst)
    if src_port is None or dst_port is None:
        print(f"Error: No physical link found between Node {src} and Node {dst} in topology.")
        return
        
    print(f"Physical link found:")
    print(f"  Node {src} (Port {src_port}) <---> Node {dst} (Port {dst_port})")
    
    # Read binary trace file
    if not os.path.exists(trace_path):
        print(f"Trace file not found: {trace_path}")
        return
        
    print(f"Scanning trace file: {trace_path}...")
    
    with open(trace_path, 'rb') as f:
        # Read SimSetting header
        len_data = f.read(4)
        if not len_data:
            print("Failed to read header.")
            return
        num_ports = struct.unpack('=I', len_data)[0]
        f.seek(4 + num_ports * 11)
        f.seek(4, 1) # Skip win
        
        # Read trace format records
        fmt = "=Q H B B I I I H B B B B x x 24s"
        record_size = 56
        
        # Direction 1: src -> dst (Dequ at src, port src_port)
        # Direction 2: dst -> src (Dequ at dst, port dst_port)
        dir1_events = []
        dir2_events = []
        
        chunk_records = 100000
        chunk_bytes = chunk_records * record_size
        records_processed = 0
        
        while True:
            data = f.read(chunk_bytes)
            if not data:
                break
            
            n_records = len(data) // record_size
            for i in range(n_records):
                offset = i * record_size
                rec = struct.unpack_from(fmt, data, offset)
                
                time_val = rec[0]
                node_val = rec[1]
                intf_val = rec[2]
                
                if node_val == src and intf_val == src_port:
                    event_val = rec[9] # 0: Recv, 1: Enqu, 2: Dequ, 3: Drop
                    size_val = rec[7]
                    qlen_val = rec[4]
                    if event_val == 2: # Dequ (Transmitted)
                        dir1_events.append((time_val, size_val, qlen_val))
                elif node_val == dst and intf_val == dst_port:
                    event_val = rec[9]
                    size_val = rec[7]
                    qlen_val = rec[4]
                    if event_val == 2: # Dequ (Transmitted)
                        dir2_events.append((time_val, size_val, qlen_val))
                        
            records_processed += n_records
            if records_processed % 10000000 == 0:
                print(f"Processed {records_processed} records...")
                
    print(f"Scan complete. Total records processed: {records_processed}")
    print(f"Found {len(dir1_events)} packets sent from {src}->{dst}")
    print(f"Found {len(dir2_events)} packets sent from {dst}->{src}")
    
    if not dir1_events and not dir2_events:
        print("No packet transmission events found for this link.")
        return
        
    # Group into time bins
    all_events = dir1_events + dir2_events
    all_events.sort(key=lambda x: x[0])
    
    start_time = all_events[0][0]
    end_time = all_events[-1][0]
    if end_time == start_time:
        end_time = start_time + 1
        
    bin_size_ns = int(bin_size_us * 1000)
    num_bins = int((end_time - start_time) / bin_size_ns) + 1
    
    bin_bytes_dir1 = [0] * num_bins
    bin_bytes_dir2 = [0] * num_bins
    
    for t, size, _ in dir1_events:
        b_idx = int((t - start_time) / bin_size_ns)
        if b_idx >= num_bins: b_idx = num_bins - 1
        bin_bytes_dir1[b_idx] += size
        
    for t, size, _ in dir2_events:
        b_idx = int((t - start_time) / bin_size_ns)
        if b_idx >= num_bins: b_idx = num_bins - 1
        bin_bytes_dir2[b_idx] += size
        
    # Convert bins to time list (ms) and bandwidth list (Gbps)
    time_series = [ (start_time + i * bin_size_ns) / 1e6 for i in range(num_bins) ]
    bw_dir1 = [ (bytes_val * 8.0) / bin_size_ns for bytes_val in bin_bytes_dir1 ]
    bw_dir2 = [ (bytes_val * 8.0) / bin_size_ns for bytes_val in bin_bytes_dir2 ]
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, bw_dir1, label=f"Node {src} -> Node {dst}", color="#3b82f6", linewidth=1.5)
    plt.plot(time_series, bw_dir2, label=f"Node {dst} -> Node {src}", color="#ec4899", linewidth=1.5)
    
    plt.title(f"Link Bandwidth Usage: Node {src} <---> Node {dst}", fontsize=14, fontweight="bold")
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Throughput (Gbps)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    
    # Save image
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()
    
    print(f"Bandwidth figure successfully saved to: {output_img}")
    
    # Output peak bandwidths
    print(f"Peak Bandwidth {src}->{dst}: {max(bw_dir1):.3f} Gbps")
    print(f"Peak Bandwidth {dst}->{src}: {max(bw_dir2):.3f} Gbps")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python3 plot_link_bandwidth.py <trace_file> <topo_file> <src_node> <dst_node> [bin_size_us] [output_img]")
        sys.exit(1)
        
    trace_f = sys.argv[1]
    topo_f = sys.argv[2]
    src_n = int(sys.argv[3])
    dst_n = int(sys.argv[4])
    
    bin_sz = 100.0
    if len(sys.argv) >= 6:
        bin_sz = float(sys.argv[5])
        
    out_img = "link_bandwidth.png"
    if len(sys.argv) >= 7:
        out_img = sys.argv[6]
        
    analyze_and_plot(trace_f, topo_f, src_n, dst_n, bin_size_us=bin_sz, output_img=out_img)
