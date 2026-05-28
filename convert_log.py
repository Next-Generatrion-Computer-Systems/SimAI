#!/usr/bin/env python3
"""
Convert SimAI Analytical_Sim.log into a structured CSV-like text format
suitable for custom network replay simulators.
"""

import re
import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Convert SimAI flow logs to a structured CSV format.")
    parser.add_argument(
        "-i", "--input", 
        default="Analytical_Sim.log", 
        help="Path to the input Analytical_Sim.log (default: Analytical_Sim.log)"
    )
    parser.add_argument(
        "-o", "--output", 
        default="gpu_flows_parsed.txt", 
        help="Path to the output txt file (default: gpu_flows_parsed.txt)"
    )
    return parser.parse_args()

def convert_log(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input log file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Parsing '{input_path}'...")
    
    # Matches: " 0,  0,  0 to  1 current_flow_id 0 prev rank:  7 parent_flow_id:  -1 child_flow_id:  9 chunk_id:  0 flow_size: 262144 chunk_count:  14 "
    flow_pattern = re.compile(
        r"^\s*(?P<channel_id>\d+),\s*"
        r"(?P<flow_idx>\d+),\s*"
        r"(?P<src>\d+)\s+to\s+"
        r"(?P<dst>\d+)\s+"
        r"current_flow_id\s+(?P<flow_id>\d+)\s+"
        r"prev rank:\s+(?P<prev_rank>-?\d+)\s+"
        r"parent_flow_id:\s+(?P<parent_flow_id>-?\d+)\s+"
        r"child_flow_id:\s+(?P<child_flow_id>-?\d+)\s+"
        r"chunk_id:\s+(?P<chunk_id>\d+)\s+"
        r"flow_size:\s+(?P<flow_size>\d+)\s+"
        r"chunk_count:\s+(?P<chunk_count>\d+)"
    )
    
    records = []
    
    with open(input_path, "r", encoding="utf-8", errors="ignore") as infile:
        for line in infile:
            # Strip log timestamp and thread prefix e.g., "[2026-05-28 00:56:50][DEBUG] [743c0e089740] "
            prefix_end = line.find("] ")
            if prefix_end != -1:
                # Find the second bracket prefix end if present
                prefix_end = line.find("] ", prefix_end + 2)
                content = line[prefix_end + 2:]
            else:
                content = line
                
            match = flow_pattern.search(content)
            if match:
                data = match.groupdict()
                records.append({
                    "flow_id": int(data["flow_id"]),
                    "src": int(data["src"]),
                    "dst": int(data["dst"]),
                    "size": int(data["flow_size"]),
                    "channel_id": int(data["channel_id"]),
                    "chunk_id": int(data["chunk_id"]),
                    "parent_flow_id": int(data["parent_flow_id"]),
                    "child_flow_id": int(data["child_flow_id"])
                })
                
    if not records:
        print("Warning: No matching flow model entries found in log file.", file=sys.stderr)
        
    print(f"Found {len(records)} flow entries. Writing to '{output_path}'...")
    
    # Write structured CSV-like file
    with open(output_path, "w", encoding="utf-8") as outfile:
        # CSV Header
        outfile.write("flow_id,src,dst,size,channel_id,chunk_id,parent_flow_id,child_flow_id\n")
        for rec in records:
            outfile.write(
                f"{rec['flow_id']},{rec['src']},{rec['dst']},{rec['size']},"
                f"{rec['channel_id']},{rec['chunk_id']},{rec['parent_flow_id']},{rec['child_flow_id']}\n"
            )
            
    print("Conversion complete!")

if __name__ == "__main__":
    args = parse_args()
    convert_log(args.input, args.output)
