

# algorithm 1 from paper in section 4.3 Poison Reconstruction i think?
# unsure what to call this file? could just call it inference maybe?
class Inference:
    def __init__(self, model, T):
        self.model = model
        self.T = T

    def forward(self, I):
        ...

        # 1. reconstruction 

        # 2. entropy cut-off + classification (poison or clean)

        # 3. poison subtraction

        # should return classification results (labels), reconstructed clean images

