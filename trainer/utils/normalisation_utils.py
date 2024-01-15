def normalize(x, min, max):
    return (x - min) / (max - min)


def denormalize(x, min, max):
    return x * (max - min) + min
