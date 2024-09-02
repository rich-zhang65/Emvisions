def get_circle_dim(x, y, w, h):
    center = (x + w // 2, y + h // 2)
    radius = max(w, h) // 2
    return center, radius
    