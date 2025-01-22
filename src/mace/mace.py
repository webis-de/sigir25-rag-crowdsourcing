import numpy as np
import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from typing import Optional, Dict, Tuple, Iterator
from scipy.special import digamma
from scipy import stats
from tqdm import trange


def get_majority_vote(annotations):
    """
    Consolidate annotations by finding the majority label for each sample.

    Parameters:
    -----------
    annotations : numpy.ndarray
        Array of shape [n_samples, n_annotators]
        Valid labels are integers, missing annotations are indicated by -1

    Returns:
    --------
    numpy.ndarray
        Array of majority vote labels of shape [n_samples]
        If no clear majority, returns -1
    """
    # Create a mask to exclude missing annotations
    valid_mask = annotations != -1

    # Allocate the result array
    majority_labels = np.full(annotations.shape[0], -1, dtype=int)

    for i in range(annotations.shape[0]):
        # Get valid labels for this sample
        sample_labels = annotations[i][valid_mask[i]]

        # If no valid labels, keep as -1
        if len(sample_labels) == 0:
            continue

        mode_result = stats.mode(sample_labels)
        # If there's a unique mode, use it
        if isinstance(mode_result.mode, np.int64):
            majority_labels[i] = mode_result.mode
        else:
            majority_labels[i] = mode_result.mode[0]

    return majority_labels


class MACE(BaseEstimator, ClassifierMixin):
    """
    Multi-Annotator Competence Estimation (MACE) Classifier.

    This model estimates the true labels from multiple annotators by modeling
    annotator reliability and spamming behavior using a probabilistic approach.

    Parameters
    ----------
    n_iter : int, default=50
        Maximum number of Expectation-Maximization (EM) iterations.

    n_restarts : int, default=100
        Number of random restarts to find the best model configuration.

    theta_prior_alpha : float, default=0.5
        Prior parameter alpha for annotator trustworthiness (Beta distribution).

    theta_prior_beta : float, default=0.5
        Prior parameter beta for annotator trustworthiness (Beta distribution).

    strategy_prior : float, default=10.0
        Dirichlet prior parameter for annotator labeling strategy.

    smoothing : float, optional, default=None
        Smoothing parameter added to counts to prevent zero probabilities.
        If None, set to 0.1 divided by the number of classes.

    Attributes
    ----------
    spam_prob_ : ndarray of shape (n_annotators, 2)
        Estimated probabilities of annotator spamming behavior.

    strat_prob_ : ndarray of shape (n_annotators, n_classes)
        Estimated annotator strategy probabilities.

    labels_ : ndarray of shape (n_instances, n_classes)
        Estimated probabilities of true labels for each instance.

    log_likelihood_ : float
        Log-likelihood of the best model configuration.

    References
    ----------
    Hovy, D., Berg-Kirkpatrick, T., Vaswani, A., & Hovy, E. (2013).
    Learning whom to trust with MACE. In: Proceedings of the 2013
    Conference of the North American Chapter of the Association for
    Computational Linguistics (pp. 1120-1130).
    """

    def __init__(
        self,
        n_iter: Optional[int] = 50,
        n_restarts: Optional[int] = 100,
        theta_prior_alpha: Optional[float] = 0.5,
        theta_prior_beta: Optional[float] = 0.5,
        strategy_prior: Optional[float] = 10.0,
        smoothing: Optional[float] = None,
        verbose: Optional[bool] = False
    ):
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.theta_prior_alpha = theta_prior_alpha
        self.theta_prior_beta = theta_prior_beta
        self.strategy_prior = strategy_prior
        self.smoothing = smoothing
        self.verbose = verbose

    def _preprocess_data(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess input annotation data.

        Parameters
        ----------
        X : ndarray of shape (n_instances, n_annotators)
            Input annotation matrix where -1 indicates missing annotations.

        Returns
        -------
        annotations : ndarray
            Processed annotation matrix.
        annotators : ndarray
            Matrix indicating which annotators provided annotations.

        Raises
        ------
        ValueError
            If input is not a 2D array.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array of annotations")

        # Compute dimensions
        self.n_samples_ = X.shape[0]
        self.n_annotators_ = X.shape[1]
        self.n_labels_ = len(np.unique(X[X >= 0]))

        # Set smoothing parameter
        self.smoothing_ = (0.1 / self.n_labels_) if self.smoothing is None else self.smoothing

        # Create annotation and annotator matrices
        annotations = X.copy()
        annotators = np.arange(self.n_annotators_)[None, :].repeat(self.n_samples_, axis=0)
        annotators[X < 0] = -1

        return annotations, annotators

    def _init_params(self) -> None:
        """
        Initialize model parameters randomly.

        Randomly initializes:
        - Spamming probabilities
        - Annotator strategy probabilities
        - Label priors
        - Prior parameters for theta and strategy
        """
        # Randomly initialize and normalize annotator spamming probabilities
        self.spam_prob_ = np.random.random((self.n_annotators_, 2))
        self.spam_prob_ /= self.spam_prob_.sum(axis=1, keepdims=True)

        # Randomly initialize and normalize annotator strategy probabilities
        self.strat_prob_ = np.random.random((self.n_annotators_, self.n_labels_))
        self.strat_prob_ /= self.strat_prob_.sum(axis=1, keepdims=True)

        # Initialize uniform label priors
        self.label_priors_ = np.ones(self.n_labels_) / self.n_labels_

        # Initialize theta and strategy priors
        self.theta_priors_ = np.tile([self.theta_prior_alpha, self.theta_prior_beta],(self.n_annotators_, 1))
        self.strategy_priors_ = np.full((self.n_annotators_, self.n_labels_), self.strategy_prior)

    def _e_step(self, annotations: np.ndarray, annotators: np.ndarray) -> Tuple[
        float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Expectation step of the EM algorithm.

        Computes label marginals, log-likelihood, and annotator counts.

        Parameters
        ----------
        annotations : ndarray
            Annotation matrix.
        annotators : ndarray
            Matrix indicating which annotators provided annotations.

        Returns
        -------
        Tuple containing:
        - log-likelihood (float)
        - gold label marginals (ndarray)
        - knowing counts (ndarray)
        - strategy counts (ndarray)
        """
        # Initialize counts
        labels = np.zeros((self.n_samples_, self.n_labels_))
        knowing_counts = np.zeros((self.n_annotators_, 2))
        strategy_counts = np.zeros((self.n_annotators_, self.n_labels_))

        log_likelihood = 0.0

        # For each instance
        for i in range(self.n_samples_):
            valid_idx = annotators[i] != -1
            if not np.any(valid_idx):
                continue

            curr_annotators = annotators[i, valid_idx]
            curr_labels = annotations[i, valid_idx]

            # Calculate label likelihood for each possible candidate label
            label_likelihood = np.zeros(self.n_labels_)
            for candidate_label in range(self.n_labels_):
                prob = self.label_priors_[candidate_label]

                for ann, label in zip(curr_annotators, curr_labels):
                    # Probability from spamming
                    spam_prob = self.spam_prob_[ann, 0] * self.strat_prob_[ann, label]
                    # Probability from knowing (only if label matches true_label)
                    know_prob = self.spam_prob_[ann, 1] if label == candidate_label else 0
                    # Combined probability of this candidate label
                    prob *= (spam_prob + know_prob)

                label_likelihood[candidate_label] = prob

            # Normalize and update likelihood
            total_likelihood = label_likelihood.sum()
            if total_likelihood > 0:
                labels[i] = label_likelihood / total_likelihood
                log_likelihood += np.log(total_likelihood)

                # Update counts
                for ann, label in zip(curr_annotators, curr_labels):
                    for candidate_label in range(self.n_labels_):
                        # Probability of this being true label
                        true_label_prob = labels[i, candidate_label]

                        # Update knowing counts
                        if label == candidate_label:
                            spam_prob = self.spam_prob_[ann, 0] * self.strat_prob_[ann, label]
                            know_prob = self.spam_prob_[ann, 1]
                            total_prob = spam_prob + know_prob

                            knowing_counts[ann, 1] += (true_label_prob * know_prob / total_prob)
                            knowing_counts[ann, 0] += (true_label_prob * spam_prob / total_prob)
                            strategy_counts[ann, label] += (true_label_prob * spam_prob / total_prob)
                        else:
                            knowing_counts[ann, 0] += true_label_prob
                            strategy_counts[ann, label] += true_label_prob

        return log_likelihood, labels, knowing_counts, strategy_counts

    def _m_step(self, knowing_counts: np.ndarray, strategy_counts: np.ndarray) -> None:
        """
        Variational M-step of the EM algorithm.

        Updates model parameters based on accumulated counts.

        Parameters
        ----------
        knowing_counts : ndarray
            Counts related to annotator knowing/spamming behavior.
        strategy_counts : ndarray
            Counts related to annotator labeling strategy.
        """
        # Add priors and smoothing
        knowing_counts = knowing_counts + self.smoothing_ + self.theta_priors_
        strategy_counts = strategy_counts + self.smoothing_ + self.strategy_priors_

        # Update parameters
        self.spam_prob_ = np.exp(digamma(knowing_counts) - digamma(knowing_counts.sum(axis=1, keepdims=True)))
        self.strat_prob_ = np.exp(digamma(strategy_counts) - digamma(strategy_counts.sum(axis=1, keepdims=True)))

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'MACE':
        """
        Fit the MACE model to annotation data.

        Parameters
        ----------
        X : ndarray of shape (n_instances, n_annotators)
            Input annotation matrix where -1 indicates missing annotations.
        y : None
            Ignored. Kept for scikit-learn compatibility.

        Returns
        -------
        self : MACE
            Fitted estimator.
        """
        annotations, annotators = self._preprocess_data(X)

        best_likelihood = float('-inf')
        best_params = None

        # Multiple random restarts to find best configuration
        pbar = tqdm.tqdm(total=self.n_restarts * self.n_iter, leave=False, disable=not self.verbose)

        for i in range(self.n_restarts):
            pbar.set_description(f'Restart {i+1}/{self.n_restarts}')
            self._init_params()

            # EM iterations
            current_likelihood = float('-inf')

            for _ in range(self.n_iter):
                pbar.update(1)
                likelihood, labels, knowing_counts, strategy_counts = \
                    self._e_step(annotations, annotators)

                # Check for convergence
                if abs(likelihood - current_likelihood) < 1e-6:
                    break

                current_likelihood = likelihood
                self._m_step(knowing_counts, strategy_counts)

            # Update best parameters if current run is better
            if current_likelihood > best_likelihood:
                best_likelihood = current_likelihood
                best_params = (
                    self.spam_prob_.copy(),
                    self.strat_prob_.copy(),
                    labels.copy()
                )

        # Set best parameters found
        self.spam_prob_, self.strat_prob_, self.labels_ = best_params
        self.log_likelihood_ = best_likelihood
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict true labels for given annotations.

        Parameters
        ----------
        X : ndarray of shape (n_instances, n_annotators)
            Input annotation matrix where -1 indicates missing annotations.

        Returns
        -------
        ndarray of shape (n_instances,)
            Predicted true labels.
        """
        check_is_fitted(self)
        annotations, annotators = self._preprocess_data(X)
        _, labels, _, _ = self._e_step(annotations, annotators)
        return np.argmax(labels, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict label probabilities for given annotations.

        Parameters
        ----------
        X : ndarray of shape (n_instances, n_annotators)
            Input annotation matrix where -1 indicates missing annotations.

        Returns
        -------
        ndarray of shape (n_instances, n_classes)
            Predicted label probabilities.
        """
        check_is_fitted(self)
        annotations, annotators = self._preprocess_data(X)
        _, labels, _, _ = self._e_step(annotations, annotators)
        return labels

    def get_annotator_stats(self) -> Dict[str, np.ndarray]:
        """
        Retrieve annotator performance statistics.

        Returns
        -------
        Dict[str, ndarray]
            Dictionary containing:
            - 'spamming_probs': Probability of annotator spamming
            - 'strategy_probs': Annotator label distribution when spamming
        """
        check_is_fitted(self)
        return {
            'spamming_probs': self.spam_prob_,
            'strategy_probs': self.strat_prob_,
        }


if __name__ == '__main__':
    # Example usage demonstrating the MACE model
    import numpy as np
    from sklearn.metrics import accuracy_score

    # Synthetic annotation data
    X = np.array([
        [-1, 0, 0, 1, -1, 0, -1, -1, 0, -1],
        [1, -1, -1, 0, -1, 1, 0, -1, -1, 0],
        [-1, -1, 0, -1, 0, 1, -1, 0, -1, 0],
        [0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
    ])

    # Initialize and fit model
    mace = MACE(n_iter=50, n_restarts=100)
    mace.fit(X)

    # Get predictions and print results
    y_pred = mace.predict(X)
    print("Input annotations:", X)
    print("Predicted labels:", y_pred)

    # Get annotator statistics
    stats = mace.get_annotator_stats()
    print("\nSpamming Probabilities:")
    print(stats['spamming_probs'])
    print("\nStrategy Probabilities:")
    print(stats['strategy_probs'])

    # Print additional information about the model
    print("\nLog Likelihood:", mace.log_likelihood_)
    print("Gold Label Marginals:\n", mace.labels_)