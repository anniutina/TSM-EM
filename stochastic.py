import math, random

class Stochastic:
    '''
        Stochastic variable
        Attributes: 
            law - code: 0 - uniform, 1 - gauss, 2 - exponential or rectangular by default, 
            distribution characteristics: location, scale, shape
        Method: value - generates random variables with specified distibution
    '''

    def __init__(self, law=0, location=0, scale=1, shape=0):
        # distribution law
        self.law = law
        # location parameter
        self.location = location
        # scale parameter
        self.scale = scale
        # form parameter
        self.shape = shape

    def value(self):
        '''
            Returns the generated value
        '''
        if self.law == 0:
            # rectangular distribution
            return random.uniform(self.location, self.location + self.scale)
        elif self.law == 1:
            # normal distribution
            # truncate to positive values
            r = random.gauss(self.location, self.scale)
            while r <= 0:
                r = random.gauss(self.location, self.scale)
            return r
        elif self.law == 2:
            # exponential distribution
            r = random.random()
            while r == 0 or r == 1:
                r = random.random()
            return -self.scale * math.log(r)
        else:
            # rectangular distribution by default
            return random.uniform(self.location, self.location + self.scale)