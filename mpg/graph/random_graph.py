import numpy as np

graph_generator=np.random.Generator(np.random.MT19937())

def set_generation_method(method):
    global graph_generator
    graph_generator = np.random.Generator(method)
    return graph_generator

if __name__=="__main__":
    print("HII")