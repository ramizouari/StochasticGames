import os
import sys
from functools import cache

import pandas as pd


@cache
def _read_xkcd():
    """Read the xkcd color table from the file data/xkcd.txt."""
    xkcd_file = os.path.join(os.path.dirname(__file__), 'data', 'xkcd.txt')
    xkcd = pd.read_csv(xkcd_file, sep='\t', skiprows=1, header=None, names=['name', 'hex'], index_col='name')
    xkcd["rgb"]=xkcd["hex"].apply(lambda x:x[1:])\
                           .apply(lambda x: tuple(int(x[i:i + 2], 16) for i in (0, 2, 4)))
    xkcd["rgb_norm"]=xkcd["rgb"].apply(lambda x: tuple(i/255 for i in x))
    return xkcd

def xkcd_color(name: str,format:str="hex"):
    """Return the RGB color of the xkcd color with the given name."""
    xkcd = _read_xkcd()
    match format:
        case "hex":
            return xkcd.loc[name, "hex"]
        case "rgb":
            return xkcd.loc[name, "rgb"]
        case "rgb_norm":
            return xkcd.loc[name, "rgb_norm"]
        case _:
            raise RuntimeError(f"Unknown format {format}")

if __name__ == "__main__":
    print(xkcd_color("red"))

class Colour:
    def __init__(self, name:str=None,hex:str=None):
        if hex is None and name is None:
            raise RuntimeError("Either hex or name must be provided")
        if hex is not None and name is not None:
            raise RuntimeError("Only one of hex or name must be provided")
        if hex is not None:
            self.hex=hex
            self.name=hex
        elif name is not None:
            if name.startswith("#"):
                self.hex=name
                self.name=name
            else:
                self.hex = xkcd_color(name,format="hex")
                self.name = name
        self._size=3
        self._symbol="&#9632;"
    def _repr_html_(self):
        return f"<font size='+{self._size}' color='{self.hex}'> {self._symbol} </font> "

    def __str__(self):
        return f"Colour({self.name}, hex={self.hex})"


if __name__ == "__main__":
    print(Colour(name="red"))
    print(Colour(hex="#ff0000"))