class Condition:
    def __init__(self, cf=None):
        if cf is not None:
            # First initialization approach
            expression = str(cf)
            tokens = expression.split()
            self.expression = expression
            self.codeFragment = cf
            self.positions = sorted([int(tk[1:]) for tk in tokens if tk[0]=='D'])
            self.is_dc = False
        else:
            # Second initialization approach (default)
            self.expression = 'dc'
            self.positions = []
            self.codeFragment = None
            self.is_dc = True

        if self.expression == "dc":
            self._str_cache = "dc"
        else:
            self._str_cache = self.expression


    def __str__(self):
        return self._str_cache
