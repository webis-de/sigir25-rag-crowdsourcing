import numpy as np
import pandas as pd

from scipy.optimize import minimize


class BradleyTerryAggregator:
    def __init__(self, tie_margin: float = 0.05, tie_threshold: float = 0.05, regularization: float = 0.2,
                 max_iter: int = 100, log_scores=True, logit_scores=False, normalize_scores=False, cython=False):
        """
        Constructor
        :param tie_margin: score margin to declare ties
        :param regularization: regularization parameter
        :param tie_threshold: difference threshold
        :param max_iter: maximum iterations for the LL optimizer
        """
        self.margin = tie_margin
        self.log_scores = log_scores
        self.logit_scores = logit_scores
        self.threshold = tie_threshold
        self.regularization = regularization
        self.max_iter = max_iter
        self.normalize = normalize_scores

    def __str__(self):
        return "bradleyterry"

    def _infer_tie(self, p):
        if not 0 <= p <= 1:
            raise ValueError('Got invalid p of ' + str(p) + '. Expected p in Interval [0, 1]')

        if not 0 <= (self.threshold + self.margin) <= 1:
            raise ValueError('Got invalid threshold and margin of ' + str(self.threshold + self.margin) +
                             '. Expected p in Interval [0, 1]')

        if not 0 <= (self.threshold - self.margin) <= 1:
            raise ValueError('Got invalid threshold and margin of ' + str(self.threshold - self.margin) +
                             '. Expected p in Interval [0, 1]')

        if p > self.threshold + self.margin:
            return False
        elif p < self.threshold - self.margin:
            return False
        else:
            return True

    def _order_pair(self, id_a, id_b, p):
        if not 0 <= p <= 1:
            raise ValueError('Got invalid p of ' + str(p) + '. Expected p in Interval [0, 1]')
        if p >= self.threshold:
            return id_a, id_b, p
        else:
            return id_b, id_a, 1 - p

    def optimize(self, comparisons, n_samples, regularization, threshold):
        # Initialize merit vector
        merits = np.ones(shape=(n_samples,), dtype=np.double)
        # Optimize using BFGS
        res = minimize(self.__log_likelihood__, merits, (comparisons, regularization, threshold), method="BFGS")
        return res.x

    @staticmethod
    def __pfunc__(i: float, j: float, t: float) -> float:
        """
        Function to compute pairwise comparison probabilities of non-ties
        :param i: merit of the winning item
        :param j: merit of the loosing item
        :param t: difference threshold
        :return: probability of item i beating item j
        """
        p = np.exp(i) / (np.exp(i) + np.exp(j) * np.exp(t))
        return np.log10(p)

    @staticmethod
    def __tfunc__(i: float, j: float, t: float) -> float:
        """
        Function to compute pairwise comparison probabilities of ties
        :param i: merit of the winning item
        :param j: merit of the loosing item
        :param t: difference threshold
        :return: probability of item i beating item j
        """
        f1 = np.exp(i) * np.exp(j) * (np.square(np.exp(t)) - 1)
        f2 = (np.exp(i) + np.exp(j) * np.exp(t)) * (np.exp(i) * np.exp(t) + np.exp(j))
        p = f1 / f2
        return np.log10(p)

    def __rfunc__(self, i: float, l: float) -> float:
        """
        Function to compute regularized probability
        :param i: item merit
        :param l: regularization factor
        :return: value of __pfunc__ for matches with dummy item weighted by l
        """
        return l * (self.__pfunc__(i, 1, 0) + self.__pfunc__(1, i, 0))

    def __log_likelihood__(self, merits: np.ndarray, comparisons: np.ndarray, regularization: float, threshold: float) -> float:
        """
        Log-Likelihood Function
        :param merits: merit vector
        :return: log-likelihood value
        """
        k: float = 0  # Maximization sum
        # Summing Edge Probabilities
        for arg1, arg2, tie in comparisons:
            if tie:
                k += self.__tfunc__(merits[arg1], merits[arg2], threshold)
            else:
                k += self.__pfunc__(merits[arg1], merits[arg2], threshold)
        # Regularization
        for i in range(merits.shape[0]):
            k += self.__rfunc__(merits[i], regularization)
        return -1 * k

    def __call__(self, comparisons: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the aggregation and return the calculated merits.
        :param pairwise_scores: pairwise score data
        """
        #assert comparisons.columns == ["id_a", "id_b", "tie"]
        self.items = list(set(comparisons["id_a"].unique().tolist() + comparisons["id_b"].unique().tolist()))
        item_mapping = {x: i for i, x in enumerate(self.items)}
        # Mapped comparisons
        self.comparisons = []
        for _, (id_a, id_b, tie) in comparisons.iterrows():
            self.comparisons.append([
                item_mapping[id_a],
                item_mapping[id_b],
                tie
            ])
        res = self.optimize(comparisons=self.comparisons, n_samples=len(self.items), regularization=self.regularization,
                            threshold=self.threshold)

        scores = {doc_id: res[index] for doc_id, index in item_mapping.items()}
        df = pd.DataFrame(scores.items(), columns=["docno", "score"])
        if self.normalize:
            df["score"] = (df["score"] - df["score"].min()) / (df["score"].max() - df["score"].min())
            return df
        else:
            return df