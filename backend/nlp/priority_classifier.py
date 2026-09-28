class PriorityClassifier:
    @staticmethod
    def classify(score: float, high_threshold: float = 80.0, medium_threshold: float = 50.0) -> str:
        if score >= high_threshold:
            return "HIGH"
        elif score >= medium_threshold:
            return "MEDIUM"
        else:
            return "LOW"
