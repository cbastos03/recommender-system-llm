import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
import numpy as np
import seaborn as sns

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "green": GREEN}

class Tester:

    def __init__(self, predictor, data, title=None):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = len(data)
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []

    def color_for(self, error):
        return "red" if error else "green"

    @staticmethod
    def str_to_bool(s):
        return str(s).strip().lower() == "true"
    
    def run_datapoint(self, i):
        datapoint = self.data[i]
        guess = self.str_to_bool(self.predictor(datapoint))
        truth = self.str_to_bool(datapoint.serendipity)
        error = guess != truth
        color = self.color_for(error)
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40]+"..."
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.colors.append(color)
        print(f"{COLOR_MAP[color]}{i+1}: Guess: {guess} Truth: {truth} Error: {error} Item: {title}{RESET}")

    def chart(self, title):
        counts = Counter(zip(self.truths, self.guesses))
        labels = ['True Pos', 'False Pos', 'True Neg', 'False Neg']
        values = [
            counts[(True, True)],
            counts[(False, True)],
            counts[(False, False)],
            counts[(True, False)],
        ]
        colors = ['green', 'orange', 'green', 'red']

        plt.figure(figsize=(8, 5))
        plt.bar(labels, values, color=colors)
        plt.title(title)
        plt.ylabel("Count")
        plt.show()

    def report(self):
        accuracy = accuracy_score(self.truths, self.guesses)
        precision = precision_score(self.truths, self.guesses)
        recall = recall_score(self.truths, self.guesses)
        f1 = f1_score(self.truths, self.guesses)

        title = f"{self.title} — Acc: {accuracy:.2f}, Prec: {precision:.2f}, Rec: {recall:.2f}, F1: {f1:.2f}"
        self.chart(title)
        print("\nDetailed classification report:")
        print(classification_report(
            self.truths,
            self.guesses,
            labels=[False, True],  # <- força a considerar as duas classes
            target_names=["Not Serendipitous", "Serendipitous"],
            zero_division=0  # evita warnings se alguma métrica for indefinida
        ))
        self.plot_confusion_matrix()
        self.plot_metrics_bar()

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.truths, self.guesses)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Not Serendipitous", "Serendipitous"],
                    yticklabels=["Not Serendipitous", "Serendipitous"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def plot_metrics_bar(self):
        precision, recall, f1, support = precision_recall_fscore_support(self.truths, self.guesses, zero_division=0)
        labels = ["Not Serendipitous", "Serendipitous"]
        metrics = [precision, recall, f1]
        metric_names = ["Precision", "Recall", "F1-score"]
        x = np.arange(len(labels))
        width = 0.25

        plt.figure(figsize=(10, 6))
        for i, metric in enumerate(metrics):
            plt.bar(x + i * width, metric, width=width, label=metric_names[i])
            for j, val in enumerate(metric):
                plt.text(x[j] + i * width, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=9)

        plt.xticks(x + width, labels)
        plt.ylim(0, 1.1)
        plt.ylabel("Score")
        plt.title("Classification Metrics by Class")
        plt.legend()
        plt.show()

    def run(self):
        self.error = 0
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function, data):
        cls(function, data).run()