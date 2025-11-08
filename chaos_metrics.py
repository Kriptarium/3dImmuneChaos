
import numpy as np

def _embed(ts, m=3, tau=1):
    ts = np.asarray(ts, dtype=float)
    N = len(ts) - (m-1)*tau
    if N <= m+1:
        return None
    Y = np.zeros((N, m))
    for i in range(m):
        Y[:, i] = ts[i*tau:i*tau+N]
    return Y

def lyapunov_rosenstein(ts, m=3, tau=1, min_separation=10, max_iter=20):
    ts = np.asarray(ts, dtype=float)
    Y = _embed(ts, m=m, tau=tau)
    if Y is None:
        return np.nan, 0.0, 0
    N = len(Y)
    nn = np.full(N, -1, dtype=int)
    for i in range(N):
        idx = np.arange(N)
        idx = idx[np.abs(idx - i) > min_separation]
        if idx.size == 0:
            continue
        d = np.linalg.norm(Y[idx] - Y[i], axis=1)
        j = idx[np.argmin(d)]
        nn[i] = j
    curves = []
    for i in range(N):
        j = nn[i]
        if j < 0:
            continue
        L = min(max_iter, N - max(i, j) - 1)
        if L <= 3:
            continue
        dist = []
        for k in range(1, L):
            a = Y[i+k]; b = Y[j+k]
            dist.append(np.linalg.norm(a-b))
        dist = np.asarray(dist)
        if np.any(dist > 0):
            curves.append(np.log(dist[dist>0]))
    if not curves:
        return np.nan, 0.0, 0
    Lmin = min(map(len, curves))
    M = np.vstack([c[:Lmin] for c in curves])
    y = M.mean(axis=0)
    x = np.arange(1, Lmin+1)
    A = np.vstack([x, np.ones_like(x)]).T
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = coeff[0]
    yhat = A @ coeff
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else 0.0
    return float(slope), float(r2), int(len(curves))

def higuchi_fd(ts, kmax=10):
    ts = np.asarray(ts, dtype=float)
    N = len(ts)
    if N < kmax*2:
        kmax = max(5, N//3)
    L = []
    k_values = list(range(1, kmax+1))
    for k in k_values:
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            x = ts[idx]
            dist = np.sum(np.abs(np.diff(x))) * (N-1) / (len(idx)*k)
            Lk.append(dist)
        if Lk:
            L.append(np.mean(Lk))
    if len(L) < 2:
        return np.nan
    logk = np.log(1.0/np.array(k_values[:len(L)]))
    logL = np.log(np.array(L))
    A = np.vstack([logk, np.ones_like(logk)]).T
    coeff, *_ = np.linalg.lstsq(A, logL, rcond=None)
    fd = coeff[0]
    return float(fd)
