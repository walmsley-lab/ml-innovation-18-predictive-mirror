"""Proper belief scores against evaluator-only hidden-state truth."""
import math

def normalize(weights):
    if not weights or any(x<0 or not math.isfinite(x) for x in weights):
        raise ValueError("Invalid weights")
    z=sum(weights)
    if z<=0:raise ValueError("Zero probability mass")
    return [x/z for x in weights]

def posterior(prior,likelihood):
    if len(prior)!=len(likelihood):raise ValueError("State dimension mismatch")
    return normalize([p*l for p,l in zip(prior,likelihood)])

def neg_log_likelihood(probs,true_index,eps=1e-12):
    p=normalize(probs)
    if not 0<=true_index<len(p):raise ValueError("Unknown state")
    return -math.log(max(eps,p[true_index]))

def brier(probs,true_index):
    p=normalize(probs)
    return sum((v-float(i==true_index))**2 for i,v in enumerate(p))

def support_coverage(probs,true_index,level=.95):
    p=normalize(probs)
    if not 0<level<=1:raise ValueError("Invalid coverage")
    order=sorted(range(len(p)),key=lambda i:-p[i])
    included=[];mass=0.
    for i in order:
        included.append(i);mass+=p[i]
        if mass>=level:break
    return int(true_index in included),len(included)

def time_to_collapse(history,threshold=.9):
    """First cue index at which max posterior reaches confidence threshold."""
    return next((t for t,p in enumerate(history) if max(normalize(p))>=threshold),None)
