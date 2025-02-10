import numpy as np
from typing import List, Tuple

class SaddlePointFinder:
    def __init__(self, payoff_matrix: List[List[float]]):
        self.payoff_matrix = np.array(payoff_matrix)
        self.rows, self.cols = self.payoff_matrix.shape
        
        # Validate input dimensions
        if self.rows <= 1 or self.cols <= 1:
            raise ValueError("Matrix must have dimensions m × n where m > 1 and n > 1")

    def find_row_minimums(self) -> np.ndarray:
        return np.min(self.payoff_matrix, axis=1)

    def find_column_maximums(self) -> np.ndarray:
        return np.max(self.payoff_matrix, axis=0)

    def find_maximin(self) -> float:
        return np.max(self.find_row_minimums())

    def find_minimax(self) -> float:
        return np.min(self.find_column_maximums())

    def find_saddle_points(self) -> List[Tuple[int, int, float]]:
        maximin = self.find_maximin()
        minimax = self.find_minimax()
        
        # If maximin != minimax, no saddle point exists
        if maximin != minimax:
            return []
        
        saddle_points = []
        
        # Find all positions where the value equals both maximin and minimax
        for i in range(self.rows):
            for j in range(self.cols):
                value = self.payoff_matrix[i, j]
                
                # Check if this position is a row minimum and column maximum
                is_row_min = value == np.min(self.payoff_matrix[i, :])
                is_col_max = value == np.max(self.payoff_matrix[:, j])
                
                if is_row_min and is_col_max:
                    saddle_points.append((i, j, value))
        
        return saddle_points

    def analyze_game(self) -> str:
        saddle_points = self.find_saddle_points()
        maximin = self.find_maximin()
        minimax = self.find_minimax()
        
        report = ["Game Analysis:"]
        report.append(f"Maximin value (Player A's guaranteed minimum): {maximin}")
        report.append(f"Minimax value (Player B's guaranteed maximum loss): {minimax}")
        
        if not saddle_points:
            report.append("\nNo saddle points found - this game has no pure strategy equilibrium.")
        else:
            report.append(f"\nFound {len(saddle_points)} saddle point(s):")
            for row, col, value in saddle_points:
                report.append(f"- Position ({row}, {col}) with value {value}")
            report.append("\nThis game has a pure strategy equilibrium.")
        
        return "\n".join(report)

def main():
    # Example usage
    payoff_matrix = [
        [3, 2, 4],
        [1, 4, 2],
        [2, 3, 1]
    ]
    
    try:
        game = SaddlePointFinder(payoff_matrix)
        print("Payoff Matrix:")
        print(game.payoff_matrix)
        print("\n" + game.analyze_game())
        
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()