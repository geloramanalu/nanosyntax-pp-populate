class K:
    def apply(self, ground):
        print(f"eigenplace({ground})")
        return ground
    
class AxPart:
    def apply(self, region):
        print(f"subpart({region})")
        return region
    
class Loc:
    def apply(self, axpart_region):
        print(f"vector_space({axpart_region})")
        return axpart_region

class Deg:
    def apply(self, vector_space):
        print(f"region_from_vector({vector_space})")
        return vector_space
    
class Place:
    # Defining constructor
    def __init__(self, ground):
        self.ground = ground
        self.k = K()
        self.axpart = AxPart()
        self.loc = Loc()
        self.deg = Deg()
        
    def derive(self):
        eigen = self.k.apply(self.ground)
        subregion = self.axpart.apply(eigen)
        vector_space = self.loc.apply(subregion)
        region = self.deg.apply(vector_space)
        return region
    
# example usage
ground = "wall"
locative_expression = Place(ground).derive()
print(f"Derived region: {locative_expression}")

