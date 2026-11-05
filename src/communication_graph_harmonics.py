"""Graph-Laplacian spectra of ACTUAL message edges versus display-grid edges.

Use the combinatorial Laplacian on undirected graphs; preserve the
constant eigenmode and tied eigenspaces. Spectral projections operate
on node states, not on their apparent positions in an RGB image.
"""
import numpy as np


def lattice_adjacency(rows,columns):
    if min(rows,columns)<2:
        raise ValueError("Grid must have >=2 rows/columns")
    n=rows*columns
    graph=np.zeros((n,n),np.float64)
    for y in range(rows):
        for x in range(columns):
            i=y*columns+x
            for dy,dx in ((1,0),(0,1)):
                py,px=y+dy,x+dx
                if py<rows and px<columns:
                    j=py*columns+px
                    graph[i,j]=graph[j,i]=1.
    return graph


def adjacency_from_edges(count,edges):
    if count<3:
        raise ValueError("Graph needs >=3 nodes")
    a=np.zeros((count,count),np.float64)
    for i,j in edges:
        if i==j or not 0<=i<count or not 0<=j<count:
            raise ValueError("Invalid graph edge")
        a[i,j]=a[j,i]=1.
    return a


def validate(a):
    a=np.asarray(a,np.float64)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or len(a)<3:
        raise ValueError("Need square graph")
    if not np.isfinite(a).all() or not np.allclose(a,a.T):
        raise ValueError("Expected finite undirected weights")
    if np.any(a<0) or not np.allclose(np.diag(a),0):
        raise ValueError("No self edges or negative weights")
    return a


def laplacian(a):
    a=validate(a)
    return np.diag(a.sum(axis=1))-a


def eigenbasis(a,count=3):
    """Return lowest nonzero modes; include entire tie at selection cutoff."""
    matrix=laplacian(a)
    values,vectors=np.linalg.eigh(matrix)
    nonzero=np.flatnonzero(values>1e-9)
    if count<1 or not len(nonzero):
        raise ValueError("Need nontrivial connected signal")
    threshold=values[nonzero[min(count,len(nonzero))-1]]
    selected=np.flatnonzero((values>1e-9)&(values<=threshold+1e-8))
    return values[selected],vectors[:,selected].T


def rewired(a,seed=0,swaps=30):
    """Simple-graph double-edge swaps preserving each node's degree exactly."""
    graph=validate(a).copy()
    if not np.array_equal(graph,(graph>0).astype(float)):
        raise ValueError("Rewiring requires binary adjacency")
    reference_degree=graph.sum(axis=0)
    rng=np.random.default_rng(int(seed))
    edges=[tuple(map(int,edge)) for edge in np.argwhere(
        np.triu(graph,1)>0)]
    if len(edges)<2:
        raise ValueError("Insufficient edges to rewire")
    for _ in range(max(0,swaps)*10):
        i,j=rng.choice(len(edges),2,replace=False)
        a,b=edges[i]
        c,d=edges[j]
        if rng.random()<.5:
            c,d=d,c
        if len({a,b,c,d})<4:
            continue
        x=tuple(sorted((a,d)))
        y=tuple(sorted((c,b)))
        if x==y or x in edges or y in edges:
            continue
        graph[a,b]=graph[b,a]=0.
        graph[c,d]=graph[d,c]=0.
        graph[x[0],x[1]]=graph[x[1],x[0]]=1.
        graph[y[0],y[1]]=graph[y[1],y[0]]=1.
        edges[i],edges[j]=x,y
        swaps-=1
        if swaps==0:
            break
    if not np.array_equal(graph.sum(axis=0),reference_degree):
        raise AssertionError("Degree-preserving rewiring failed")
    return graph


def projection(state,basis):
    """Project [N,C] node activations onto [K,N] orthonormal eigenmodes."""
    state=np.asarray(state,np.float64)
    basis=np.asarray(basis,np.float64)
    if state.ndim!=2 or basis.ndim!=2 or basis.shape[1]!=len(state):
        raise ValueError("Expected node activations and compatible modes")
    centered=state-state.mean(axis=0,keepdims=True)
    return basis.T@(basis@centered)


def energy_fraction(state,basis):
    centered=state-state.mean(axis=0,keepdims=True)
    denominator=float(np.sum(centered**2))
    numerator=float(np.sum(projection(state,basis)**2))
    return numerator/max(denominator,1e-12)


def propagate(a,state,steps=5,gain=.4):
    """Fixed graph-local dynamical reference: x += gain(Px-x)."""
    a=validate(a)
    if steps<1 or not 0<=gain<=1:
        raise ValueError("Invalid dynamics")
    degrees=np.maximum(a.sum(axis=1,keepdims=True),1)
    x=np.asarray(state,np.float64).copy()
    result=[]
    for _ in range(steps):
        x=x+gain*((a@x)/degrees-x)
        result.append(x.copy())
    return np.stack(result)


def interventions(graph,state,steps=5,rank=3,seed=1):
    """Same graph dynamics, low-mode suppression vs matched-energy null."""
    graph=validate(graph)
    eig,basis=eigenbasis(graph,rank)
    random=np.random.default_rng(seed).normal(size=basis.shape[::-1])
    random-=random.mean(axis=0)
    # Random rank-k zero-mean orthonormal basis.
    null=np.linalg.qr(random,mode="reduced")[0].T[:len(basis)]
    low=projection(state,basis)
    distractor=projection(state,null)
    target_norm=np.linalg.norm(low)
    null_norm=np.linalg.norm(distractor)
    if target_norm<1e-12:
        aligned=np.zeros_like(low)
        control=np.zeros_like(low)
    else:
        aligned=low
        control=(distractor*target_norm/null_norm
                 if null_norm>1e-12 else low.copy())
    original=propagate(graph,state,steps)
    suppressed=propagate(graph,state-aligned,steps)
    matched=propagate(graph,state-control,steps)
    return {"eigenvalues":eig.tolist(),
        "low_removed_l2":float(np.linalg.norm(aligned)),
        "control_removed_l2":float(np.linalg.norm(control)),
        "reference_forecasts":original,
        "low_mode_forecasts":suppressed,
        "matched_null_forecasts":matched,
        "low_response_l2":float(np.linalg.norm(original-suppressed)),
        "null_response_l2":float(np.linalg.norm(original-matched)),
        "low_fraction":energy_fraction(state,basis)}
