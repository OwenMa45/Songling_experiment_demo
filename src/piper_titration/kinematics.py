"""Small position/orientation IK solver; also testable without a physics renderer."""
import numpy as np


def tool_rotation(heading):
    c, s = np.cos(heading), np.sin(heading)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])


def rotation_error(target, current):
    # Quaternion-free skew residual; robust away from pi, with trace-based fallback.
    r = target @ current.T
    angle = np.arccos(np.clip((np.trace(r)-1)/2, -1, 1))
    v = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])
    if angle < 1e-7:
        return v/2
    if np.pi-angle < 1e-5:
        vals, vectors = np.linalg.eig(r)
        axis = np.real(vectors[:, np.argmin(abs(vals-1))])
        return axis / np.linalg.norm(axis) * angle
    return v * angle / (2*np.sin(angle))


def solve(forward, position, rotation, limits, initial=None):
    def residual(q):
        p, r = forward(q)
        return np.r_[p-position, rotation_error(rotation, r)*0.12]
    rng = np.random.default_rng(7)
    seeds = [] if initial is None else [np.array(initial, dtype=float)]
    seeds += [limits.mean(axis=1)]
    seeds += [rng.uniform(limits[:, 0], limits[:, 1]) for _ in range(18)]
    best, best_error = None, float("inf")
    for q in seeds:
        q = np.clip(q, limits[:, 0], limits[:, 1])
        for _ in range(180):
            e = residual(q)
            norm = np.linalg.norm(e)
            if norm < best_error:
                best, best_error = q.copy(), norm
            if np.linalg.norm(e[:3]) < 0.001 and np.linalg.norm(e[3:]) < 0.002:
                return q
            jac = np.column_stack([(residual(q+np.eye(6)[i]*1e-5)-e)/1e-5 for i in range(6)])
            delta = -jac.T @ np.linalg.solve(jac@jac.T + np.eye(6)*1e-5, e)
            delta = np.clip(delta, -0.2, 0.2)
            found = False
            for scale in (1, .5, .25, .1):
                candidate = np.clip(q+delta*scale, limits[:, 0], limits[:, 1])
                if np.linalg.norm(residual(candidate)) < norm:
                    q = candidate
                    found = True
                    break
            if not found:
                break
    e = residual(best)
    if np.linalg.norm(e[:3]) > 0.004 or np.linalg.norm(e[3:]) > 0.012:
        raise ValueError(f"Unreachable pose, residual {e.tolist()}")
    return best
