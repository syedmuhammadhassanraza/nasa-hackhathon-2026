"""
Swarm Intelligence: Ant Colony Optimization (ACO) for Feature Selection
Optimizes physiological feature selection from WESAD & OSD-575 multi-modal vitals.
Employs pheromone trails, heuristic information, evaporation, and elite ant reinforcement.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class AntColonyFeatureSelector:
    def __init__(
        self,
        n_ants: int = 15,
        n_iterations: int = 20,
        alpha: float = 1.0,       # Pheromone weight
        beta: float = 1.2,        # Heuristic weight
        evaporation_rate: float = 0.15,
        q0: float = 0.7,          # Exploitation vs exploration probability
        feature_penalty: float = 0.05,
        random_state: int = 42,
    ):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = evaporation_rate
        self.q0 = q0
        self.feature_penalty = feature_penalty
        self.rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
        n_samples, n_features = X.shape
        self.feature_names = feature_names

        # 1. Compute Heuristic Information (Mutual Information / Fisher Discriminant Ratio)
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        heuristics = np.zeros(n_features)
        
        # Calculate single-feature F-statistic or correlation ratio
        classes = np.unique(y)
        overall_mean = np.mean(X_norm, axis=0)
        ss_between = np.zeros(n_features)
        ss_within = np.zeros(n_features)
        for c in classes:
            Xc = X_norm[y == c]
            nc = len(Xc)
            if nc > 0:
                ss_between += nc * (np.mean(Xc, axis=0) - overall_mean) ** 2
                ss_within += np.sum((Xc - np.mean(Xc, axis=0)) ** 2, axis=0)
        
        heuristics = (ss_between / (ss_within + 1e-6))
        # Normalize heuristic to [0.1, 1.0]
        heuristics = 0.1 + 0.9 * (heuristics - heuristics.min()) / (heuristics.max() - heuristics.min() + 1e-9)

        # 2. Initialize Pheromone matrix (vector per feature)
        pheromones = np.ones(n_features) * 1.0

        history = []
        best_fitness = -1.0
        best_subset = None
        best_metrics = {}

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        print(f"[ACO] Initiating Swarm Optimization with {self.n_ants} ants over {self.n_iterations} iterations...")

        for it in range(1, self.n_iterations + 1):
            iteration_subsets = []
            iteration_fitnesses = []

            for ant in range(self.n_ants):
                # Probabilistic subset construction
                # Probability of feature selection: P_i = (tau_i^alpha * eta_i^beta) / sum(...)
                scores = (pheromones ** self.alpha) * (heuristics ** self.beta)
                probs = scores / np.sum(scores)

                # Subset size randomly sampled between 4 and n_features-2
                subset_size = self.rng.integers(4, min(14, n_features))
                
                # Ant decides which features to visit
                selected_indices = self.rng.choice(
                    n_features, size=subset_size, replace=False, p=probs
                )
                selected_indices = np.sort(selected_indices)

                # Evaluate subset with fast RF classifier
                X_sub = X_norm[:, selected_indices]
                f1_scores = []
                acc_scores = []

                for tr_idx, va_idx in cv.split(X_sub, y):
                    clf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42)
                    clf.fit(X_sub[tr_idx], y[tr_idx])
                    preds = clf.predict(X_sub[va_idx])
                    f1_scores.append(f1_score(y[va_idx], preds, average="weighted", zero_division=0))
                    acc_scores.append(accuracy_score(y[va_idx], preds))

                mean_f1 = float(np.mean(f1_scores))
                mean_acc = float(np.mean(acc_scores))
                
                # Fitness balances performance with parsimony (minimal feature count)
                cardinality_ratio = len(selected_indices) / n_features
                fitness = mean_acc - (self.feature_penalty * cardinality_ratio)

                iteration_subsets.append(selected_indices)
                iteration_fitnesses.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_subset = selected_indices
                    best_metrics = {
                        "accuracy": mean_acc,
                        "f1_score": mean_f1,
                        "num_features": len(selected_indices),
                        "selected_features": [self.feature_names[i] for i in selected_indices],
                    }

            # Pheromone Evaporation
            pheromones = (1.0 - self.rho) * pheromones

            # Pheromone Deposit by Ants (proportional to fitness)
            for subset, fitness in zip(iteration_subsets, iteration_fitnesses):
                deposit = max(0.01, fitness)
                pheromones[subset] += deposit * 0.1

            # Elite Ant Reinforcement (extra pheromone for best global solution)
            if best_subset is not None:
                pheromones[best_subset] += 0.5 * best_fitness

            # Keep pheromones in healthy bounds [0.05, 10.0]
            pheromones = np.clip(pheromones, 0.05, 10.0)

            history.append({
                "iteration": it,
                "best_iteration_fitness": float(np.max(iteration_fitnesses)),
                "mean_iteration_fitness": float(np.mean(iteration_fitnesses)),
                "best_global_fitness": float(best_fitness),
                "active_best_features": [self.feature_names[i] for i in best_subset],
                "pheromone_levels": {name: round(float(ph), 4) for name, ph in zip(self.feature_names, pheromones)},
            })

            print(f"  Iteration {it:2d}/{self.n_iterations} | Best Fitness: {best_fitness:.4f} | Features: {len(best_subset)}")

        return {
            "best_subset_indices": best_subset.tolist(),
            "best_features": best_metrics["selected_features"],
            "best_accuracy": best_metrics["accuracy"],
            "best_f1": best_metrics["f1_score"],
            "num_features": best_metrics["num_features"],
            "all_features_count": n_features,
            "reduction_pct": round((1.0 - best_metrics["num_features"] / n_features) * 100, 1),
            "final_pheromones": {name: round(float(ph), 3) for name, ph in zip(self.feature_names, pheromones)},
            "convergence_history": history,
        }


def run_aco_pipeline(data_path: Path | str = "processed/wesad_features.csv", out_path: Path | str = "models/saved/aco_results.json"):
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c not in ["subject", "t_start_s", "t_end_s", "label", "state", "label_purity"]]
    X = df[feature_cols].values
    y = df["state"].values

    aco = AntColonyFeatureSelector(n_ants=12, n_iterations=15, evaporation_rate=0.18, random_state=42)
    results = aco.fit(X, y, feature_cols)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[ACO] Optimization Complete!")
    print(f"Optimal Feature Subset ({results['num_features']}/{results['all_features_count']} - {results['reduction_pct']}% reduction):")
    for f in results["best_features"]:
        print(f"  * {f} (Pheromone: {results['final_pheromones'].get(f, 0):.2f})")
    print(f"Validation Accuracy: {results['best_accuracy']*100:.2f}% | F1: {results['best_f1']:.3f}")
    return results


if __name__ == "__main__":
    run_aco_pipeline()
