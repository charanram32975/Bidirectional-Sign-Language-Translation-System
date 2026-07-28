class ConfidenceCheck:
    def is_confident(self, confidence, threshold=0.7):
        return confidence >= threshold
