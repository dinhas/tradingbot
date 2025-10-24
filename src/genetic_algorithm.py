import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable
import logging
from dataclasses import dataclass
import random
from src.backtest import backtest_method1_donchian, backtest_method2_atr, backtest_method3_volume
from src.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Individual:
    """Represents one individual (parameter set) in the population."""
    genes: List[float]  # Parameter values
    fitness: float = 0.0
    metrics: Dict = None

    def __repr__(self):
        return f"Individual(genes={[f'{g:.2f}' for g in self.genes]}, fitness={self.fitness:.4f})"


class GeneticAlgorithm:
    """Genetic Algorithm for parameter optimization."""

    def __init__(self,
                 param_ranges: List[Tuple[float, float]],
                 backtest_func: Callable,
                 data: pd.DataFrame,
                 population_size: int = POPULATION_SIZE,
                 generations: int = GENERATIONS,
                 crossover_rate: float = CROSSOVER_RATE,
                 mutation_rate: float = MUTATION_RATE,
                 elitism_count: int = ELITISM_COUNT):

        self.param_ranges = param_ranges
        self.backtest_func = backtest_func
        self.data = data
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count

        self.population: List[Individual] = []
        self.best_individual: Individual = None
        self.fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []

    def initialize_population(self):
        """Create initial random population."""
        logger.info(f"Initializing population of {self.population_size} individuals...")
        self.population = []

        for _ in range(self.population_size):
            genes = []
            for min_val, max_val in self.param_ranges:
                # Generate random value within range
                if isinstance(min_val, int) and isinstance(max_val, int):
                    gene = random.randint(min_val, max_val)
                else:
                    gene = random.uniform(min_val, max_val)
                genes.append(gene)

            individual = Individual(genes=genes)
            self.population.append(individual)

    def calculate_fitness(self, metrics: Dict) -> float:
        """
        Calculate fitness score from backtest metrics.

        Fitness components:
        - Win rate (35%)
        - Profit factor (30%)
        - Risk-reward ratio (20%)
        - Trade count (15%)

        Penalties:
        - Too few trades (< 30)
        - Excessive drawdown (> 30%)
        """
        if metrics['num_trades'] == 0:
            return 0.0

        # Penalties
        trade_penalty = 1.0
        if metrics['num_trades'] < MIN_TRADES_REQUIRED:
            trade_penalty = 0.5  # Heavy penalty for insufficient trades

        drawdown_penalty = 1.0
        if metrics['max_drawdown'] > MAX_DRAWDOWN_THRESHOLD:
            drawdown_penalty = 0.6  # Penalty for excessive risk

        # Normalize components to 0-1 range
        win_rate_component = metrics['win_rate']  # Already 0-1

        # Profit factor (normalize: 1.5+ is good, cap at 3.0)
        profit_factor_norm = min(metrics['profit_factor'] / 3.0, 1.0) if metrics['profit_factor'] > 0 else 0

        # Risk-reward ratio (normalize: 2.0+ is good, cap at 3.0)
        rr_norm = min(metrics['risk_reward_ratio'] / 3.0, 1.0) if metrics['risk_reward_ratio'] > 0 else 0

        # Trade count (normalize: 100+ trades is good, cap at 200)
        trade_count_norm = min(metrics['num_trades'] / 200.0, 1.0)

        # Composite fitness
        fitness = (
            win_rate_component * 0.35 +
            profit_factor_norm * 0.30 +
            rr_norm * 0.20 +
            trade_count_norm * 0.15
        ) * trade_penalty * drawdown_penalty

        return fitness

    def evaluate_population(self):
        """Evaluate fitness for all individuals in population."""
        for i, individual in enumerate(self.population):
            if individual.fitness == 0.0:  # Only evaluate if not already evaluated
                # Run backtest with this individual's parameters
                metrics = self.backtest_func(self.data, *individual.genes)
                individual.metrics = metrics
                individual.fitness = self.calculate_fitness(metrics)

        # Sort population by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Update best individual
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = self.population[0]

        # Record history
        self.fitness_history.append(self.population[0].fitness)
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        self.avg_fitness_history.append(avg_fitness)

    def tournament_selection(self) -> Individual:
        """Select an individual using tournament selection."""
        tournament = random.sample(self.population, TOURNAMENT_SIZE)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform uniform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return Individual(genes=parent1.genes.copy()), Individual(genes=parent2.genes.copy())

        child1_genes = []
        child2_genes = []

        for g1, g2 in zip(parent1.genes, parent2.genes):
            if random.random() < 0.5:
                child1_genes.append(g1)
                child2_genes.append(g2)
            else:
                child1_genes.append(g2)
                child2_genes.append(g1)

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def mutate(self, individual: Individual):
        """Apply Gaussian mutation to individual's genes."""
        for i, (min_val, max_val) in enumerate(self.param_ranges):
            if random.random() < self.mutation_rate:
                # Gaussian mutation with 20% of range as std dev
                std_dev = (max_val - min_val) * 0.2
                mutation = random.gauss(0, std_dev)
                individual.genes[i] += mutation

                # Clip to valid range
                individual.genes[i] = max(min_val, min(max_val, individual.genes[i]))

                # Round integers
                if isinstance(min_val, int) and isinstance(max_val, int):
                    individual.genes[i] = int(round(individual.genes[i]))

    def evolve_generation(self):
        """Create next generation through selection, crossover, and mutation."""
        new_population = []

        # Elitism: Keep best individuals
        new_population.extend(self.population[:self.elitism_count])

        # Generate rest of population
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Crossover
            child1, child2 = self.crossover(parent1, parent2)

            # Mutation
            self.mutate(child1)
            self.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population[:self.population_size]

    def run(self) -> Dict:
        """Run the genetic algorithm for specified generations."""
        logger.info("="*80)
        logger.info("Starting Genetic Algorithm Optimization")
        logger.info("="*80)

        # Initialize
        self.initialize_population()

        # Initial evaluation
        logger.info("Evaluating initial population...")
        self.evaluate_population()
        logger.info(f"Generation 0: Best Fitness = {self.population[0].fitness:.4f}, "
                   f"Avg Fitness = {self.avg_fitness_history[-1]:.4f}")

        # Evolution loop
        for gen in range(1, self.generations + 1):
            self.evolve_generation()
            self.evaluate_population()

            if gen % 10 == 0:
                logger.info(f"Generation {gen}: Best Fitness = {self.population[0].fitness:.4f}, "
                          f"Avg Fitness = {self.avg_fitness_history[-1]:.4f}, "
                          f"Best Params = {[f'{g:.2f}' for g in self.population[0].genes]}")

        logger.info("="*80)
        logger.info("Optimization Complete!")
        logger.info(f"Best Fitness: {self.best_individual.fitness:.4f}")
        logger.info(f"Best Parameters: {self.best_individual.genes}")
        logger.info("="*80)

        return {
            'best_individual': self.best_individual,
            'best_fitness': self.best_individual.fitness,
            'best_params': self.best_individual.genes,
            'best_metrics': self.best_individual.metrics,
            'fitness_history': self.fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'final_population': self.population
        }


def optimize_method1(data: pd.DataFrame, generations: int = GENERATIONS) -> Dict:
    """Optimize Donchian Channel method."""
    logger.info("\n🔍 OPTIMIZING METHOD 1: Donchian Channel Breakout")

    param_ranges = [
        (DONCHIAN_PERIOD_MIN, DONCHIAN_PERIOD_MAX)  # lookback_period
    ]

    ga = GeneticAlgorithm(
        param_ranges=param_ranges,
        backtest_func=backtest_method1_donchian,
        data=data,
        generations=generations
    )

    result = ga.run()
    result['method'] = 'Donchian Channel'
    return result


def optimize_method2(data: pd.DataFrame, generations: int = GENERATIONS) -> Dict:
    """Optimize ATR Volatility method."""
    logger.info("\n🔍 OPTIMIZING METHOD 2: ATR Volatility Breakout")

    param_ranges = [
        (ATR_PERIOD_MIN, ATR_PERIOD_MAX),           # atr_period
        (ATR_MULTIPLIER_MIN, ATR_MULTIPLIER_MAX)    # atr_multiplier
    ]

    ga = GeneticAlgorithm(
        param_ranges=param_ranges,
        backtest_func=backtest_method2_atr,
        data=data,
        generations=generations
    )

    result = ga.run()
    result['method'] = 'ATR Volatility'
    return result


def optimize_method3(data: pd.DataFrame, generations: int = GENERATIONS) -> Dict:
    """Optimize Volume-Confirmed method."""
    logger.info("\n🔍 OPTIMIZING METHOD 3: Volume-Confirmed Breakout")

    param_ranges = [
        (VOLUME_LOOKBACK_MIN, VOLUME_LOOKBACK_MAX),      # lookback_period
        (VOLUME_THRESHOLD_MIN, VOLUME_THRESHOLD_MAX),    # volume_threshold
        (VOLUME_MA_PERIOD_MIN, VOLUME_MA_PERIOD_MAX)     # volume_ma_period
    ]

    ga = GeneticAlgorithm(
        param_ranges=param_ranges,
        backtest_func=backtest_method3_volume,
        data=data,
        generations=generations
    )

    result = ga.run()
    result['method'] = 'Volume-Confirmed'
    return result
