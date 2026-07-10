import numpy as np
from numba import jit, njit, prange, set_num_threads, config
from numba.typed import Dict
from numba import types
from random import sample
import pandas as pd
import random
import time
from datasets import Dataset, DatasetDict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

@njit
def perform_operation(op_code, x, y):
    match op_code:
        case _ if y is None:    return np.nan
        case 0:                 return x + y                                                            # Add
        case 1:                 return x - y                                                            # Min
        case 2:                 return x * y                                                            # Mul
        case 3:                 return x / y if y != 0 else np.nan                                      # Div
        case 4:                 return x // y if y != 0 else np.nan                                     # Floor Div
        case 5:                 return x % y if y != 0 else np.nan                                      # Mod
        case 6:                 return max(x, y)                                                        # Max
        case 7:                 return min(x, y)                                                        # Min
        case 8:                 return abs(x)                                                           # Abs
        case 9:                 return -x                                                               # Neg
        case 10:                return np.sign(x)                                                       # Sign
        case 11:                return round(x)                                                         # Round
        case 12:                return np.ceil(x)                                                       # Ceil
        case 13:                return np.floor(x)                                                      # Floor
        case 14:                return np.trunc(x)                                                      # Trunc
        case 17:                return ~x if isinstance(x, int) else np.nan                             # Is
        case 15:                return x & y if isinstance(x, int) and isinstance(y, int) else np.nan   
        case 16:                return x | y if isinstance(x, int) and isinstance(y, int) else np.nan
        case 18:                return x ^ y if isinstance(x, int) and isinstance(y, int) else np.nan
        case 19:                return x << y if isinstance(x, int) and isinstance(y, int) else np.nan
        case 20:                return x >> y if isinstance(x, int) and isinstance(y, int) else np.nan
        case 21:                return bool(x) and bool(y)                                              # AND
        case 22:                return bool(x) or bool(y)                                               # OR
        case 23:                return not bool(x)                                                      # NOT
        case 24:                return bool(x) ^ bool(y)                                                # XOR
        case 25:                return not (bool(x) and bool(y))                                        # NAND
        case 26:                return not (bool(x) or bool(y))                                         # NOR
        case 27:                return x == y                                                           # Eq
        case 28:                return x != y                                                           # Neq
        case 29:                return x < y                                                            # Less
        case 30:                return x <= y                                                           # Lesseq
        case 31:                return x > y                                                            # Grt
        case 32:                return x >= y                                                           # Grteq
    return np.nan

# ========================================
# COMPLEX EXPRESSION GENERATION CLASSES
# ========================================

@dataclass
class ComplexityConfig:
    """Configuration for complex expression generation"""
    max_depth: int = 3
    complexity_freq: float = 0.3
    group_freq: float = 0.5
    max_intermediate_value: int = 10000
    operator_weights: Dict[str, float] = None
    precedence_aware: bool = True
    
    def __post_init__(self):
        if self.operator_weights is None:
            self.operator_weights = {'+': 0.3, '-': 0.3, '*': 0.25, '/': 0.15}

class ExpressionNode(ABC):
    """Abstract base class for expression tree nodes"""
    
    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate the expression and return the result"""
        pass
    
    @abstractmethod
    def to_string(self, parent_precedence: int = 0) -> str:
        """Convert to string with precedence-aware parentheses"""
        pass
    
    @abstractmethod
    def get_depth(self) -> int:
        """Get the maximum depth of this subtree"""
        pass
    
    @abstractmethod
    def get_operator_count(self) -> int:
        """Get the total number of operators in this subtree"""
        pass

class NumberNode(ExpressionNode):
    """Leaf node containing a numeric value"""
    
    def __init__(self, value: float):
        self.value = value
    
    def evaluate(self) -> float:
        return self.value
    
    def to_string(self, parent_precedence: int = 0) -> str:
        # Format numbers nicely - integers without decimal point
        if self.value == int(self.value):
            return str(int(self.value))
        else:
            return f"{self.value:g}"
    
    def get_depth(self) -> int:
        return 0
    
    def get_operator_count(self) -> int:
        return 0

class BinaryOpNode(ExpressionNode):
    """Internal node representing a binary operation"""
    
    # Operator precedence levels (higher number = higher precedence)
    PRECEDENCE = {
        '+': 1, '-': 1,
        '*': 2, '/': 2, '//': 2, '%': 2,
        'max': 3, 'min': 3
    }
    
    def __init__(self, operator: str, left: ExpressionNode, right: ExpressionNode):
        self.operator = operator
        self.left = left
        self.right = right
        self.precedence = self.PRECEDENCE.get(operator, 0)
    
    def evaluate(self) -> float:
        """Evaluate the expression, checking for bounds and invalid operations"""
        left_val = self.left.evaluate()
        right_val = self.right.evaluate()
        
        if np.isnan(left_val) or np.isnan(right_val):
            return np.nan
            
        # Map operators to op_codes for reusing existing perform_operation
        op_map = {'+': 0, '-': 1, '*': 2, '/': 3, '//': 4, '%': 5, 'max': 6, 'min': 7}
        
        if self.operator in op_map:
            result = perform_operation(op_map[self.operator], left_val, right_val)
            return result
        else:
            return np.nan
    
    def to_string(self, parent_precedence: int = 0) -> str:
        """Convert to string with precedence-aware parentheses"""
        # Get child strings with appropriate precedence context
        left_str = self.left.to_string(self.precedence)
        
        # For right operand, we need to be more careful about precedence
        # If right operand has same precedence and we're dealing with - or /, add parentheses
        right_precedence = self.precedence
        if self.operator in ['-', '/'] and isinstance(self.right, BinaryOpNode):
            if self.right.precedence <= self.precedence:
                right_precedence = self.precedence + 1
        
        right_str = self.right.to_string(right_precedence)
        
        expression = f"{left_str} {self.operator} {right_str}"
        
        # Add parentheses if this operation has lower precedence than parent
        # OR if we want to make groupings more explicit
        if (self.precedence < parent_precedence or
            (parent_precedence > 0 and self.precedence <= parent_precedence)):
            expression = f"({expression})"
            
        return expression
    
    def get_depth(self) -> int:
        return 1 + max(self.left.get_depth(), self.right.get_depth())
    
    def get_operator_count(self) -> int:
        return 1 + self.left.get_operator_count() + self.right.get_operator_count()

class ExpressionTree:
    """Container for managing expression trees"""
    
    def __init__(self, root: ExpressionNode, config: ComplexityConfig):
        self.root = root
        self.config = config
    
    def evaluate(self) -> float:
        """Evaluate the entire expression"""
        return self.root.evaluate()
    
    def to_string(self) -> str:
        """Convert to string representation"""
        return self.root.to_string()
    
    def get_complexity(self) -> int:
        """Get the complexity (depth) of the expression"""
        return self.root.get_depth()
    
    def get_operator_count(self) -> int:
        """Get total number of operators"""
        return self.root.get_operator_count()
    
    def has_groups(self) -> bool:
        """Check if expression has parentheses (groupings)"""
        return self.get_complexity() > 1
    
    def validate_bounds(self) -> bool:
        """Check if all intermediate values are within bounds"""
        result = self.evaluate()
        return not np.isnan(result) and abs(result) <= self.config.max_intermediate_value

class ExpressionGenerator:
    """Generates complex arithmetic expressions"""
    
    def __init__(self, config: ComplexityConfig):
        self.config = config
        
    def generate_expression_tree(self, min_val: int, max_val: int,
                                current_depth: int = 0) -> ExpressionNode:
        """Generate a complex expression tree with better hierarchical structure"""
        
        # Base case: generate a number if at max depth
        if current_depth >= self.config.max_depth:
            return NumberNode(random.randint(min_val, max_val))
        
        # At intermediate depths, decide whether to create a subtree or stop
        # Higher probability of continuing at lower depths
        continue_prob = self.config.group_freq * (1.0 - current_depth / self.config.max_depth)
        if random.random() > continue_prob and current_depth > 0:
            return NumberNode(random.randint(min_val, max_val))
        
        # Generate a binary operation with different strategies
        operator = self._select_operator()
        
        # Strategy 1: Create mixed precedence operations to force parentheses
        if current_depth == 0 and random.random() < 0.5:
            # Create a structure like (a op1 b) op2 (c op3 d)
            left_op = self._select_different_precedence_operator(operator)
            right_op = self._select_different_precedence_operator(operator)
            
            left_left = NumberNode(random.randint(min_val, max_val))
            left_right = NumberNode(random.randint(min_val, max_val))
            left = BinaryOpNode(left_op, left_left, left_right)
            
            right_left = NumberNode(random.randint(min_val, max_val))
            right_right = NumberNode(random.randint(min_val, max_val))
            right = BinaryOpNode(right_op, right_left, right_right)
        else:
            # Standard recursive generation
            left = self.generate_expression_tree(min_val, max_val, current_depth + 1)
            right = self.generate_expression_tree(min_val, max_val, current_depth + 1)
        
        return BinaryOpNode(operator, left, right)
    
    def _select_different_precedence_operator(self, current_op: str) -> str:
        """Select an operator with different precedence to force parentheses"""
        current_precedence = BinaryOpNode.PRECEDENCE.get(current_op, 0)
        
        # Get operators with different precedence
        different_ops = [op for op, prec in BinaryOpNode.PRECEDENCE.items()
                        if prec != current_precedence and op in self.config.operator_weights]
        
        if different_ops:
            return random.choice(different_ops)
        else:
            return self._select_operator()
    
    def _select_operator(self) -> str:
        """Select an operator based on configured weights"""
        operators = list(self.config.operator_weights.keys())
        weights = list(self.config.operator_weights.values())
        return random.choices(operators, weights=weights)[0]
    
    def generate_valid_expression(self, min_val: int, max_val: int,
                                 max_attempts: int = 100) -> Optional[ExpressionTree]:
        """Generate a valid expression within bounds"""
        for _ in range(max_attempts):
            root = self.generate_expression_tree(min_val, max_val)
            tree = ExpressionTree(root, self.config)
            
            if tree.validate_bounds():
                return tree
                
        return None  # Failed to generate valid expression

# Set threads and verify
set_num_threads(1)
print(f"Numba threads configured: {config.NUMBA_NUM_THREADS}")

@jit(nopython=True, parallel=True)
def generate_samples(min_val, max_val, step, n, op_code):
    """
    Numba-optimized function to generate samples
    """
    # Convert n to integer
    n_samples = int(n)
    
    # Pre-allocate arrays without dtype specification
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    results = np.zeros(n_samples)
    
    # Generate values in parallel
    for i in prange(n_samples):
        x[i] = np.random.randint(min_val, max_val + 1)
        y[i] = np.random.randint(min_val, max_val + 1)
        results[i] = perform_operation(op_code, x[i], y[i])
        
    return x, y, results

def gen_simple_arithmetic_sample(min_val, max_val, step, n, operator):
    """
    Generator function that uses Numba-optimized functions
    """
    # Map operators to op_codes
    op_map = {'+': 0, '-': 1, '*': 2, '/': 3, '//': 4, '%': 5, 'max': 6, 'min': 7, 'abs': 8, 'neg': 9,
        'sign': 10, 'round': 11, 'ceil': 12, 'floor': 13, 'trunc': 14, 'NaN': 17, 'and': 21, 'or': 22,
        'not': 23, 'xor': 24, 'nand': 25, 'nor': 26, 'eq': 27, 'neq': 28, 'less': 29, 'lesseq': 30, 'gtr': 31, 'gtreq': 32}
    
    op_code = op_map[operator]
    
    # Generate all samples at once using Numba function
    x_vals, y_vals, results = generate_samples(min_val, max_val, step, n, op_code)
    
    # Yield results one at a time
    for i in range(n):
        yield {
            "x": float(x_vals[i]),
            "y": float(y_vals[i]),
            "z": float(results[i]) if not np.isnan(results[i]) else None,
            "text": f"{float(x_vals[i]):g} {operator} {float(y_vals[i]):g} = {float(results[i]) if not np.isnan(results[i]) else None:g}",
            "operator": operator,
        }

def gen_complex_arithmetic_sample(min_val: int, max_val: int, step: int, n: int,
                                 complexity_config: ComplexityConfig) -> List[Dict[str, Any]]:
    """
    Generate complex arithmetic expressions with configurable complexity
    """
    generator = ExpressionGenerator(complexity_config)
    samples = []
    
    for i in range(n):
        # Decide whether to generate simple or complex expression
        use_complex = random.random() < complexity_config.complexity_freq
        
        if use_complex:
            # Generate complex expression
            tree = generator.generate_valid_expression(min_val, max_val)
            
            if tree is not None:
                expression_str = tree.to_string()
                result = tree.evaluate()
                
                # Create intermediate values list for debugging
                intermediate_values = [result]  # Could be enhanced to track all intermediate steps
                
                sample = {
                    "expression": expression_str,
                    "result": float(result) if not np.isnan(result) else None,
                    "complexity": tree.get_complexity(),
                    "operator_count": tree.get_operator_count(),
                    "has_groups": tree.has_groups(),
                    "text": f"{expression_str} = {float(result):g}" if not np.isnan(result) else f"{expression_str} = None",
                    "intermediate_values": intermediate_values,
                    "is_complex": True
                }
                samples.append(sample)
            else:
                # Fallback to simple expression if complex generation failed
                fallback_samples = list(gen_simple_arithmetic_sample(min_val, max_val, step, 1, '+'))
                if fallback_samples:
                    fallback_sample = fallback_samples[0]
                    fallback_sample.update({
                        "expression": f"{fallback_sample['x']:g} + {fallback_sample['y']:g}",
                        "result": fallback_sample["z"],
                        "complexity": 1,
                        "operator_count": 1,
                        "has_groups": False,
                        "intermediate_values": [fallback_sample["z"]],
                        "is_complex": False
                    })
                    samples.append(fallback_sample)
        else:
            # Generate simple expression
            operators = list(complexity_config.operator_weights.keys())
            operator = random.choice(operators)
            
            simple_samples = list(gen_simple_arithmetic_sample(min_val, max_val, step, 1, operator))
            if simple_samples:
                simple_sample = simple_samples[0]
                simple_sample.update({
                    "expression": f"{simple_sample['x']:g} {operator} {simple_sample['y']:g}",
                    "result": simple_sample["z"],
                    "complexity": 1,
                    "operator_count": 1,
                    "has_groups": False,
                    "intermediate_values": [simple_sample["z"]],
                    "is_complex": False
                })
                samples.append(simple_sample)
    
    return samples

def gen_sample_batch(width = 0xF, n = 10, d = 100, val_split = 1/10,
                    use_complex: bool = False, complexity_config: Optional[ComplexityConfig] = None):
    """
    Enhanced batch generation with support for complex expressions
    """
    train_samples = []
    val_samples = []
    
    # Initialize complexity config if using complex expressions
    if use_complex and complexity_config is None:
        complexity_config = ComplexityConfig()
    
    m = n
    f = d

    while m < width:
        start_time = time.time()

        if use_complex and complexity_config is not None:
            # Generate complex expressions
            try:
                current_samples = gen_complex_arithmetic_sample(1, int(m), 1, int(f), complexity_config)
                val_samples.extend(current_samples[:int(val_split*len(current_samples))])
                train_samples.extend(current_samples[int(val_split*len(current_samples)):])
            except Exception as e:
                print(f"Error generating complex samples at n={n}, m={m}, d={d}, f={f}: {str(e)}")
                break
        else:
            # Generate simple expressions (original behavior)
            ops = ['+', '-', '*', '/', '//', '%']  # Define ops for simple expressions
            for o in ops:
                try:
                    current_samples = list(gen_simple_arithmetic_sample(1, int(m), 1, int(f), o))
                    val_samples.extend(current_samples[:int(val_split*f)])
                    train_samples.extend(current_samples[int(val_split*f):])
                except Exception as e:
                    print(f"Error at n={n}, m={m}, d={d}, f={f}, op={o}: {str(e)}")
                    break

        ops_count = 6 if not use_complex else 1  # 6 operators for simple, 1 batch for complex
        print(f"Generated {int(f*ops_count) if not use_complex else int(f)} samples ({len(train_samples)} training, {len(val_samples)} validation) each for range 0-{m} ({len(train_samples) + len(val_samples) / (time.time() - start_time):.2f} per second)")
        m = m*n
        f = f*d

    print(f"{len(train_samples) + len(val_samples)} examples in {time.time() - start_time:.2f} seconds ({len(train_samples) + len(val_samples) / (time.time() - start_time):.2f} per second)")
    
    # Determine columns based on whether complex expressions were used
    if use_complex:
        # Use extended schema for complex expressions
        columns = ['expression', 'result', 'complexity', 'operator_count', 'has_groups', 'text', 'intermediate_values', 'is_complex']
        train_df = pd.DataFrame(train_samples)
        val_df = pd.DataFrame(val_samples)
    else:
        # Use original schema for simple expressions
        columns = ['x', 'y', 'z', 'operator', 'text']
        train_df = pd.DataFrame(train_samples, columns=columns)
        val_df = pd.DataFrame(val_samples, columns=columns)
    
    # Drop duplicates only on hashable columns (exclude list columns like intermediate_values)
    hashable_columns = [col for col in train_df.columns if not train_df[col].apply(lambda x: isinstance(x, list)).any()]
    train_df.drop_duplicates(subset=hashable_columns, inplace=True)
    val_df.drop_duplicates(subset=hashable_columns, inplace=True)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
    })

    return dataset_dict

def test_complex_expressions():
    """Test function to validate complex expression generation"""
    print("=== Testing Complex Expression Generation ===")
    
    # Test configuration
    config = ComplexityConfig(
        max_depth=2,
        complexity_freq=1.0,  # Always generate complex
        group_freq=0.8,
        max_intermediate_value=1000,
        operator_weights={'+': 0.4, '-': 0.3, '*': 0.2, '/': 0.1}
    )
    
    # Generate a few test expressions
    test_samples = gen_complex_arithmetic_sample(1, 10, 1, 5, config)
    
    print("Generated test expressions:")
    for i, sample in enumerate(test_samples):
        print(f"{i+1}. {sample['text']}")
        print(f"   Expression: {sample['expression']}")
        print(f"   Complexity: {sample['complexity']}, Operators: {sample['operator_count']}")
        print(f"   Has Groups: {sample['has_groups']}, Is Complex: {sample['is_complex']}")
        print()
    
    return len(test_samples) > 0

if __name__ == "__main__":
    start_time = time.time()

    random.seed(42)

    ops = ['+', '-', '*', '/', '//', '%'] #, 'and', 'nand', 'or', 'nor', 'xor']
    #'max', 'min', 'abs', 'neg', 'sign', 'round', 'ceil', 'floor', 'trunc', 'NaN', 'not', 'avg',
    _ = generate_samples(0, 10, 1, 10, 0)  # Warmup

    width = {
        "smol": 0xFF,
        "mid": 0xFFF,
        "big": 0xFFFF,
    }

    # Example: Generate simple arithmetic (original behavior)
    print("=== Generating Simple Arithmetic Dataset ===")
    ds_smol_simple = gen_sample_batch(width["smol"])
    ds_smol_simple.save_to_disk("./scratch/arithmetic0_smol_simple")
    
    # Example: Generate complex arithmetic with default configuration
    print("\n=== Generating Complex Arithmetic Dataset ===")
    complex_config = ComplexityConfig(
        max_depth=3,
        complexity_freq=0.6,
        group_freq=0.7,
        max_intermediate_value=5000,
        operator_weights={'+': 0.25, '-': 0.25, '*': 0.3, '/': 0.2}
    )
    
    ds_smol_complex = gen_sample_batch(
        width=width["smol"],
        d=300,
        use_complex=True,
        complexity_config=complex_config
    )
    ds_smol_complex.save_to_disk("./scratch/arithmetic0_smol_complex")
    
    # Example: Generate a few sample expressions to show the output
    print("\n=== Sample Complex Expressions ===")
    sample_expressions = gen_complex_arithmetic_sample(1, 20, 1, 10, complex_config)
    for i, expr in enumerate(sample_expressions[:5]):
        print(f"Example {i+1}: {expr['text']}")
        print(f"  Complexity: {expr['complexity']}, Operators: {expr['operator_count']}, Has Groups: {expr['has_groups']}")
    
    #ds_smol.save_to_disk("/tmp/guac/arithmetic0_smol")
    #ds_mid = gen_sample_batch(width["mid"])
    #ds_mid.save_to_disk("./scratch/arithmetic0_mid")
    #ds_big = gen_sample_batch(width["big"])
    #ds_big.save_to_disk("./scratch/arithmetic0_big")


def generate_complex_arithmetic_dataset(
    complexity_level: str = "moderate",
    size: str = "small",
    save_path: Optional[str] = None
) -> DatasetDict:
    """
    Convenience function to generate complex arithmetic datasets with preset configurations
    
    Args:
        complexity_level: "simple", "moderate", "complex", or "mixed"
        size: "small", "medium", "large" 
        save_path: Optional path to save the dataset
    
    Returns:
        DatasetDict with train/validation splits
    """
    
    # Complexity configurations
    complexity_configs = {
        "simple": ComplexityConfig(
            max_depth=1,
            complexity_freq=0.0,  # Only simple expressions
            group_freq=0.0,
            operator_weights={'+': 0.4, '-': 0.4, '*': 0.15, '/': 0.05}
        ),
        "moderate": ComplexityConfig(
            max_depth=3,
            complexity_freq=0.6,
            group_freq=0.7,
            max_intermediate_value=5000,
            operator_weights={'+': 0.25, '-': 0.25, '*': 0.3, '/': 0.2}
        ),
        "complex": ComplexityConfig(
            max_depth=4,
            complexity_freq=0.8,
            group_freq=0.9,
            max_intermediate_value=10000,
            operator_weights={'+': 0.2, '-': 0.2, '*': 0.3, '/': 0.3}
        ),
        "mixed": ComplexityConfig(
            max_depth=3,
            complexity_freq=0.5,
            group_freq=0.6,
            max_intermediate_value=5000,
            operator_weights={'+': 0.3, '-': 0.3, '*': 0.25, '/': 0.15}
        )
    }
    
    # Size configurations
    size_configs = {
        "small": {"width": 0xFF, "n": 10, "d": 50},
        "medium": {"width": 0xFFF, "n": 10, "d": 100},
        "large": {"width": 0xFFFF, "n": 10, "d": 200}
    }
    
    config = complexity_configs.get(complexity_level, complexity_configs["moderate"])
    size_params = size_configs.get(size, size_configs["small"])
    
    use_complex = complexity_level != "simple"
    
    dataset = gen_sample_batch(
        width=size_params["width"],
        n=size_params["n"],
        d=size_params["d"],
        use_complex=use_complex,
        complexity_config=config
    )
    
    if save_path:
        dataset.save_to_disk(save_path)
        print(f"Dataset saved to: {save_path}")
    
    return dataset
