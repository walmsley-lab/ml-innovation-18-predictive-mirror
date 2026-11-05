"""Toy physical-placement model; explicit logical hops and buffer limits."""
import math

def positions(count,columns):
    if count<1 or columns<1:raise ValueError("Invalid topology size")
    return [(i//columns,i%columns) for i in range(count)]

def hop_distance(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def map_communication(nodes,edges,columns=4,channels=4,word_bytes=4,
                      local_capacity_bytes=1024,remote_weight=1.,
                      offchip_weight=20.):
    if min(nodes,columns,channels,word_bytes,local_capacity_bytes)<1:
        raise ValueError("Invalid physical assumptions")
    coords=positions(nodes,columns)
    state_bytes=nodes*channels*word_bytes
    routed_bytes=0;weighted_hop_bytes=0
    for i,j in edges:
        if not 0<=i<nodes or not 0<=j<nodes:
            raise ValueError("Invalid logical edge")
        hops=hop_distance(coords[i],coords[j])
        if hops:
            routed_bytes+=channels*word_bytes
            weighted_hop_bytes+=hops*channels*word_bytes
    spill_bytes=max(0,state_bytes-local_capacity_bytes)
    return {"positions":coords,"state_bytes":state_bytes,
            "remote_message_bytes":routed_bytes,
            "weighted_hop_bytes":weighted_hop_bytes,
            "offchip_spill_floor_bytes":spill_bytes,
            "estimated_cost":remote_weight*weighted_hop_bytes+
                offchip_weight*spill_bytes,
            "credibility":"analytic_not_hardware_measured"}

def compare_placements(nodes,edges,widths=(2,4,8),**kwargs):
    return {w:map_communication(nodes,edges,columns=w,**kwargs) for w in widths}
