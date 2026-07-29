import sys
import struct
import os

def parse_trace(trace_path, target_node, target_port, bin_size_ns=100000):
    if not os.path.exists(trace_path):
        print(f"Trace file not found: {trace_path}")
        return

    print(f"Parsing trace file: {trace_path} (size: {os.path.getsize(trace_path)/1e9:.2f} GB)...")
    
    with open(trace_path, 'rb') as f:
        # 1. Read SimSetting Header
        len_data = f.read(4)
        if not len_data:
            print("Failed to read header length.")
            return
        num_ports = struct.unpack('=I', len_data)[0]
        
        # Skip port speeds
        # Each entry is: node (2B) + intf (1B) + bps (8B) = 11B
        f.seek(4 + num_ports * 11)
        # Skip win (4B)
        f.seek(4, 1)
        
        # 2. Read TraceFormat Records
        # Struct format: 
        # time (Q: 8B), node (H: 2B), intf (B: 1B), qidx (B: 1B), qlen (I: 4B)
        # sip (I: 4B), dip (I: 4B), size (H: 2B), l3Prot (B: 1B), event (B: 1B)
        # ecn (B: 1B), nodeType (B: 1B), padding (2B), union (24B) = 56B
        fmt = "=Q H B B I I I H B B B B x x 24s"
        record_size = 56
        
        # Accumulators
        max_qlen = 0
        events = []
        
        chunk_records = 100000
        chunk_bytes = chunk_records * record_size
        
        print("Scanning records (this may take a short moment due to file size)...")
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
                
                if node_val == target_node and intf_val == target_port:
                    qidx_val = rec[3]
                    qlen_val = rec[4]
                    sip_val = rec[5]
                    dip_val = rec[6]
                    size_val = rec[7]
                    event_val = rec[9] # 0: Recv, 1: Enqu, 2: Dequ, 3: Drop
                    
                    events.append((time_val, event_val, size_val, qlen_val))
                    if qlen_val > max_qlen:
                        max_qlen = qlen_val
            
            records_processed += n_records
            if records_processed % 10000000 == 0:
                print(f"Processed {records_processed} records...")

    print(f"Scan complete. Total records processed: {records_processed}")
    print(f"Found {len(events)} events matching Node {target_node}, Port {target_port}")
    
    if not events:
        print("No events found for this interface. Is tracing enabled on this node?")
        return

    # Sort events by time just in case
    events.sort(key=lambda x: x[0])
    
    # Group into bins for bandwidth calculation
    start_time = events[0][0]
    end_time = events[-1][0]
    
    if end_time == start_time:
        end_time = start_time + 1
        
    num_bins = int((end_time - start_time) / bin_size_ns) + 1
    bin_bytes = [0] * num_bins
    bin_max_qlen = [0] * num_bins
    
    for t, event_type, size, qlen in events:
        bin_idx = int((t - start_time) / bin_size_ns)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
            
        # Accumulate tx bytes on Dequeue events
        if event_type == 2: # Dequ (transmitted packet)
            bin_bytes[bin_idx] += size
            
        if qlen > bin_max_qlen[bin_idx]:
            bin_max_qlen[bin_idx] = qlen

    print("\n" + "="*60)
    print(f"LINK ANALYSIS REPORT FOR Node {target_node} Port {target_port}")
    print("="*60)
    print(f"Max Queue Length observed: {max_qlen} bytes ({max_qlen/1024:.2f} KB)")
    print(f"Time Range: {start_time/1e6:.4f} ms to {end_time/1e6:.4f} ms")
    print(f"Bin size: {bin_size_ns/1000:.1f} us")
    print("-"*60)
    print(f"{'Time (ms)':<15}{'Max Qlen (KB)':<18}{'Bandwidth (Gbps)':<20}")
    print("-"*60)
    
    # Print non-zero bins to keep output clean, but show the timeline
    printed_rows = 0
    for i in range(num_bins):
        bin_time_ms = (start_time + i * bin_size_ns) / 1e6
        bin_qlen_kb = bin_max_qlen[i] / 1024.0
        # bandwidth in Gbps = (bytes * 8) / (bin_size_ns * 1e-9 * 1e9) = bytes * 8 / bin_size_ns
        bin_bw_gbps = (bin_bytes[i] * 8.0) / bin_size_ns
        
        # Only print if there is traffic or queue occupancy, or to show timeline boundary
        if bin_bytes[i] > 0 or bin_max_qlen[i] > 0 or i == 0 or i == num_bins - 1:
            print(f"{bin_time_ms:<15.4f}{bin_qlen_kb:<18.2f}{bin_bw_gbps:<20.3f}")
            printed_rows += 1
            if printed_rows > 100:
                print("... [truncated, too many busy periods] ...")
                break
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 analyze_link.py <trace_file> <node_id> <port_id> [bin_size_us]")
        sys.exit(1)
        
    trace_file = sys.argv[1]
    node = int(sys.argv[2])
    port = int(sys.argv[3])
    bin_us = 100 # default 100 us
    if len(sys.argv) >= 5:
        bin_us = float(sys.argv[4])
        
    parse_trace(trace_file, node, port, bin_size_ns=int(bin_us * 1000))
