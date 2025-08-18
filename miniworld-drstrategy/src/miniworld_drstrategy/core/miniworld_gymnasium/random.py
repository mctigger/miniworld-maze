import numpy as np
from gymnasium.utils import seeding

class RandGen:
    """
    Random value generator
    """

    def __init__(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)

    def random_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.integers(low, high)

    def random_float(self, low, high, shape=None):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high, size=shape)

    def random_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.integers(0, 2) == 0)

    def choice(self, iterable, probs=None):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self.np_random.choice(len(lst), p=probs)
        return lst[idx]

    def random_color(self):
        """
        Pick a random color name
        """

        from .entities.base_entity import COLOR_NAMES
        return self.choice(COLOR_NAMES)

    def subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self.choice(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    # Backward compatibility aliases
    def int(self, low, high):
        """Legacy alias for random_int (deprecated: shadows built-in)"""
        return self.random_int(low, high)
    
    def float(self, low, high, shape=None):
        """Legacy alias for random_float (deprecated: shadows built-in)"""
        return self.random_float(low, high, shape)
    
    def bool(self):
        """Legacy alias for random_bool (deprecated: shadows built-in)"""
        return self.random_bool()
        
    def color(self):
        """Legacy alias for random_color (deprecated: not descriptive)"""
        return self.random_color()
