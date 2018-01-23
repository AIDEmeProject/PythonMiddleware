
def get_generator_average(gen):
    sum, count = 0, 0
    for x in gen:
        sum, count = sum + x, count + 1
    return [] if count == 0 else sum / count