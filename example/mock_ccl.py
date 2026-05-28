import sys
import math

class MockCCLParser:
    def __init__(self, tp_size=1, ep_size=1, pp_size=1, dp_size=1, all_gpus=1):
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.all_gpus = all_gpus
        
        self.g_flow_id = 0
        
    def generate_ring_allgather(self, group_size, data_size):
        # Simplification: Assume 1 ring channel, chunksize = data_size / group_size
        if group_size <= 1:
            return []
            
        chunk_size = math.ceil(data_size / group_size)
        chunk_count = group_size - 1
        
        flows = []
        for chunk_id in range(chunk_count):
            for rank in range(group_size):
                src = rank
                dst = (rank + 1) % group_size
                flows.append({
                    "flow_id": self.g_flow_id,
                    "src": src,
                    "dst": dst,
                    "chunk_id": chunk_id,
                    "flow_size": chunk_size,
                    "type": "ALLGATHER_RING"
                })
                self.g_flow_id += 1
        return flows

    def generate_ring_reducescatter(self, group_size, data_size):
        # Similar to AllGather in terms of chunking
        if group_size <= 1:
            return []
            
        chunk_size = math.ceil(data_size / group_size)
        chunk_count = group_size - 1
        
        flows = []
        for chunk_id in range(chunk_count):
            for rank in range(group_size):
                src = rank
                dst = (rank + 1) % group_size
                flows.append({
                    "flow_id": self.g_flow_id,
                    "src": src,
                    "dst": dst,
                    "chunk_id": chunk_id,
                    "flow_size": chunk_size,
                    "type": "REDUCESCATTER_RING"
                })
                self.g_flow_id += 1
        return flows

    def generate_alltoall(self, group_size, data_size):
        if group_size <= 1:
            return []
            
        chunk_size = math.ceil(data_size / group_size)
        
        flows = []
        for i in range(group_size):
            for j in range(group_size):
                if i == j:
                    continue
                flows.append({
                    "flow_id": self.g_flow_id,
                    "src": i,
                    "dst": j,
                    "chunk_id": 0,
                    "flow_size": chunk_size,
                    "type": "ALLTOALL"
                })
                self.g_flow_id += 1
        return flows

    def process_comm(self, comm_type, comm_size):
        if comm_size == 0 or comm_type == "NONE":
            return []
            
        # Simplified mapping:
        # If the operator is an EP specific operator, use EP group. Otherwise default to TP group.
        # Note: In real Astra-Sim, mapping of operators to groups (TP/DP/EP) is much more complex
        # and depends on the specific forward/backward pass logic of Hybrid Parallelism.
        if "EP" in comm_type:
            group_size = self.ep_size
        elif comm_type in ["ALLGATHER", "REDUCESCATTER"]:
            group_size = self.tp_size # often TP for column/row linear
        else:
            group_size = self.tp_size
            
        if "ALLGATHER" in comm_type:
            return self.generate_ring_allgather(group_size, comm_size)
        elif "REDUCESCATTER" in comm_type:
            return self.generate_ring_reducescatter(group_size, comm_size)
        elif "ALLTOALL" in comm_type:
            return self.generate_alltoall(group_size, comm_size)
        return []

    def parse_workload(self, filepath):
        results = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Parse first line for topology sizes
        if "HYBRID_TRANSFORMER" in lines[0]:
            parts = lines[0].strip().split()
            for i, part in enumerate(parts):
                if part == "model_parallel_NPU_group:": self.tp_size = int(parts[i+1])
                elif part == "ep:": self.ep_size = int(parts[i+1])
                elif part == "pp:": self.pp_size = int(parts[i+1])
                elif part == "all_gpus:": self.all_gpus = int(parts[i+1])
        
        print(f"Parsed Topology: TP={self.tp_size}, EP={self.ep_size}, PP={self.pp_size}, TOTAL={self.all_gpus}")
        
        # skip metadata line
        for line_idx, line in enumerate(lines[2:], start=3):
            parts = line.strip().split()
            if len(parts) < 8: continue
            
            layer_name = parts[0]
            fwd_comm_type = parts[3]
            fwd_comm_size = int(parts[4])
            bwd_comm_type = parts[6]
            bwd_comm_size = int(parts[7])
            
            fwd_flows = self.process_comm(fwd_comm_type, fwd_comm_size)
            for flow in fwd_flows:
                results.append(f"Line {line_idx} [{layer_name}] [FWD] src: {flow['src']}, dst: {flow['dst']}, chunk_id: {flow['chunk_id']}, flow_size: {flow['flow_size']} ({flow['type']})")
                
            bwd_flows = self.process_comm(bwd_comm_type, bwd_comm_size)
            for flow in bwd_flows:
                results.append(f"Line {line_idx} [{layer_name}] [BWD] src: {flow['src']}, dst: {flow['dst']}, chunk_id: {flow['chunk_id']}, flow_size: {flow['flow_size']} ({flow['type']})")
                
        return results

if __name__ == "__main__":
    parser = MockCCLParser()
    out = parser.parse_workload("workload.txt")
    with open("mock_ccl_flows.log", "w") as f:
        for line in out:
            f.write(line + "\n")
    print(f"Generated {len(out)} flows and saved to mock_ccl_flows.log")
