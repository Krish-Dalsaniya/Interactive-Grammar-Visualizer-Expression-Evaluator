# import streamlit as st
# import pandas as pd
# import re
# import graphviz
# import networkx as nx
# from collections import defaultdict, deque
# from typing import Dict, Set, List, Tuple, Optional, Union
# import matplotlib.pyplot as plt
# from io import StringIO

# # Set page config
# st.set_page_config(
#     page_title="Grammar Visualizer & Expression Evaluator",
#     page_icon="📝",
#     layout="wide"
# )

# class GrammarParser:
#     """Handles context-free grammar parsing and analysis"""
    
#     def __init__(self):
#         self.rules = defaultdict(list)
#         self.non_terminals = set()
#         self.terminals = set()
#         self.start_symbol = None
        
#     def parse_grammar(self, grammar_text: str) -> bool:
#         """Parse grammar rules from text input"""
#         try:
#             self.rules.clear()
#             self.non_terminals.clear()
#             self.terminals.clear()
            
#             lines = [line.strip() for line in grammar_text.strip().split('\n') if line.strip()]
            
#             if not lines:
#                 return False
                
#             self.start_symbol = lines[0].split('->')[0].strip()
            
#             for line in lines:
#                 if '->' not in line:
#                     continue
                    
#                 left, right = line.split('->', 1)
#                 left = left.strip()
                
#                 if not left:
#                     continue
                    
#                 self.non_terminals.add(left)
                
#                 # Split productions by |
#                 productions = [p.strip() for p in right.split('|') if p.strip()]
                
#                 for prod in productions:
#                     symbols = prod.split()
#                     self.rules[left].append(symbols)
                    
#                     # Identify terminals and non-terminals
#                     for symbol in symbols:
#                         if symbol.isupper() or len(symbol) == 1:
#                             if symbol not in self.non_terminals:
#                                 self.terminals.add(symbol)
#                         else:
#                             if symbol not in ['epsilon', 'ε']:
#                                 self.terminals.add(symbol)
            
#             return True
            
#         except Exception as e:
#             st.error(f"Error parsing grammar: {str(e)}")
#             return False
    
#     def compute_first_sets(self) -> Dict[str, Set[str]]:
#         """Compute FIRST sets for all non-terminals"""
#         first = defaultdict(set)
        
#         # Initialize
#         for nt in self.non_terminals:
#             first[nt] = set()
            
#         changed = True
#         while changed:
#             changed = False
            
#             for nt in self.non_terminals:
#                 old_size = len(first[nt])
                
#                 for production in self.rules[nt]:
#                     if not production or production == ['epsilon'] or production == ['ε']:
#                         first[nt].add('ε')
#                         continue
                        
#                     for i, symbol in enumerate(production):
#                         if symbol in self.terminals:
#                             first[nt].add(symbol)
#                             break
#                         elif symbol in self.non_terminals:
#                             first[nt].update(first[symbol] - {'ε'})
#                             if 'ε' not in first[symbol]:
#                                 break
                        
#                         # If we've processed all symbols and all had epsilon
#                         if i == len(production) - 1:
#                             first[nt].add('ε')
                
#                 if len(first[nt]) > old_size:
#                     changed = True
                    
#         return dict(first)
    
#     def compute_follow_sets(self, first_sets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
#         """Compute FOLLOW sets for all non-terminals"""
#         follow = defaultdict(set)
        
#         # Initialize
#         for nt in self.non_terminals:
#             follow[nt] = set()
            
#         # Start symbol gets $
#         if self.start_symbol:
#             follow[self.start_symbol].add('$')
            
#         changed = True
#         while changed:
#             changed = False
            
#             for nt in self.non_terminals:
#                 for production in self.rules[nt]:
#                     for i, symbol in enumerate(production):
#                         if symbol in self.non_terminals:
#                             old_size = len(follow[symbol])
                            
#                             # Look at symbols after current symbol
#                             beta = production[i + 1:]
                            
#                             if not beta:  # Symbol is at end
#                                 follow[symbol].update(follow[nt])
#                             else:
#                                 # Compute FIRST(beta)
#                                 first_beta = set()
#                                 for j, beta_symbol in enumerate(beta):
#                                     if beta_symbol in self.terminals:
#                                         first_beta.add(beta_symbol)
#                                         break
#                                     elif beta_symbol in self.non_terminals:
#                                         first_beta.update(first_sets[beta_symbol] - {'ε'})
#                                         if 'ε' not in first_sets[beta_symbol]:
#                                             break
                                    
#                                     # If all symbols in beta can derive epsilon
#                                     if j == len(beta) - 1:
#                                         first_beta.add('ε')
                                
#                                 follow[symbol].update(first_beta - {'ε'})
                                
#                                 if 'ε' in first_beta:
#                                     follow[symbol].update(follow[nt])
                            
#                             if len(follow[symbol]) > old_size:
#                                 changed = True
                                
#         return dict(follow)

# class ExpressionParser:
#     """Handles arithmetic expression parsing"""
    
#     def __init__(self):
#         self.tokens = []
#         self.pos = 0
        
#     def tokenize(self, expression: str) -> List[Tuple[str, str]]:
#         """Tokenize arithmetic expression"""
#         tokens = []
#         i = 0
        
#         while i < len(expression):
#             if expression[i].isspace():
#                 i += 1
#                 continue
                
#             if expression[i].isdigit():
#                 num = ''
#                 while i < len(expression) and (expression[i].isdigit() or expression[i] == '.'):
#                     num += expression[i]
#                     i += 1
#                 tokens.append(('NUMBER', num))
#             elif expression[i].isalpha():
#                 var = ''
#                 while i < len(expression) and expression[i].isalnum():
#                     var += expression[i]
#                     i += 1
#                 tokens.append(('ID', var))
#             elif expression[i] in '+-*/()':
#                 tokens.append(('OP', expression[i]))
#                 i += 1
#             else:
#                 raise ValueError(f"Invalid character: {expression[i]}")
                
#         tokens.append(('EOF', ''))
#         return tokens
    
#     def recursive_descent_parse(self, expression: str) -> Tuple[any, str]:
#         """Parse using recursive descent (top-down)"""
#         try:
#             self.tokens = self.tokenize(expression)
#             self.pos = 0
            
#             result = self.parse_expression()
            
#             if self.current_token()[0] != 'EOF':
#                 raise ValueError("Unexpected tokens at end of expression")
                
#             # Create parse tree visualization
#             tree_dot = self.create_parse_tree_dot(result)
            
#             return result, tree_dot
            
#         except Exception as e:
#             raise ValueError(f"Parse error: {str(e)}")
    
#     def current_token(self):
#         """Get current token"""
#         if self.pos < len(self.tokens):
#             return self.tokens[self.pos]
#         return ('EOF', '')
    
#     def consume_token(self, expected_type=None):
#         """Consume current token"""
#         token = self.current_token()
#         if expected_type and token[0] != expected_type:
#             raise ValueError(f"Expected {expected_type}, got {token[0]}")
#         self.pos += 1
#         return token
    
#     def parse_expression(self):
#         """Parse expression: term (('+' | '-') term)*"""
#         node = {'type': 'expression', 'children': []}
#         left = self.parse_term()
#         node['children'].append(left)
        
#         while self.current_token()[1] in ['+', '-']:
#             op = self.consume_token('OP')
#             right = self.parse_term()
#             node['children'].extend([{'type': 'operator', 'value': op[1]}, right])
            
#         return node
    
#     def parse_term(self):
#         """Parse term: factor (('*' | '/') factor)*"""
#         node = {'type': 'term', 'children': []}
#         left = self.parse_factor()
#         node['children'].append(left)
        
#         while self.current_token()[1] in ['*', '/']:
#             op = self.consume_token('OP')
#             right = self.parse_factor()
#             node['children'].extend([{'type': 'operator', 'value': op[1]}, right])
            
#         return node
    
#     def parse_factor(self):
#         """Parse factor: number | id | '(' expression ')'"""
#         token = self.current_token()
        
#         if token[0] == 'NUMBER':
#             self.consume_token('NUMBER')
#             return {'type': 'number', 'value': token[1]}
#         elif token[0] == 'ID':
#             self.consume_token('ID')
#             return {'type': 'id', 'value': token[1]}
#         elif token[1] == '(':
#             self.consume_token('OP')
#             expr = self.parse_expression()
#             self.consume_token('OP')  # closing )
#             return {'type': 'parentheses', 'children': [expr]}
#         else:
#             raise ValueError(f"Unexpected token: {token}")
    
#     def create_parse_tree_dot(self, node, counter=[0]) -> str:
#         """Create DOT representation of parse tree"""
#         dot = graphviz.Digraph()
#         dot.attr(rankdir='TB')
        
#         def add_node(n, parent_id=None):
#             node_id = f"node_{counter[0]}"
#             counter[0] += 1
            
#             if n['type'] == 'number':
#                 dot.node(node_id, f"NUMBER\\n{n['value']}", shape='box', style='filled', fillcolor='lightblue')
#             elif n['type'] == 'id':
#                 dot.node(node_id, f"ID\\n{n['value']}", shape='box', style='filled', fillcolor='lightgreen')
#             elif n['type'] == 'operator':
#                 dot.node(node_id, f"OP\\n{n['value']}", shape='diamond', style='filled', fillcolor='orange')
#             else:
#                 dot.node(node_id, n['type'].upper(), shape='ellipse', style='filled', fillcolor='lightgray')
            
#             if parent_id:
#                 dot.edge(parent_id, node_id)
            
#             if 'children' in n:
#                 for child in n['children']:
#                     add_node(child, node_id)
                    
#             return node_id
        
#         add_node(node)
#         return dot.source
    
#     def evaluate_expression(self, expression: str) -> float:
#         """Evaluate arithmetic expression"""
#         try:
#             # Simple evaluation using Python's eval (for demonstration)
#             # In production, you'd want a safer evaluator
#             safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
#             return eval(safe_expr)
#         except:
#             return None

# def create_sample_grammar():
#     """Create sample grammar for testing"""
#     return """E -> E + T | T
# T -> T * F | F
# F -> ( E ) | id"""

# def create_sample_expression():
#     """Create sample expression for testing"""
#     return "3 + 4 * (5 - 2)"

# def main():
#     st.title("🔤 Interactive Grammar Visualizer & Expression Evaluator")
#     st.markdown("---")
    
#     # Create tabs
#     tab1, tab2 = st.tabs(["📝 Grammar Visualizer", "🧮 Expression Evaluator"])
    
#     with tab1:
#         st.header("Context-Free Grammar Analyzer")
        
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             st.subheader("Grammar Input")
            
#             # Sample grammar button
#             if st.button("Load Sample Grammar", key="sample_grammar"):
#                 st.session_state.grammar_text = create_sample_grammar()
            
#             grammar_text = st.text_area(
#                 "Enter grammar rules (one per line):",
#                 value=st.session_state.get('grammar_text', ''),
#                 height=200,
#                 help="Format: A -> B C | D\nUse 'epsilon' or 'ε' for empty production"
#             )
            
#             if st.button("Analyze Grammar", key="analyze"):
#                 if grammar_text.strip():
#                     parser = GrammarParser()
                    
#                     if parser.parse_grammar(grammar_text):
#                         st.success("Grammar parsed successfully!")
                        
#                         # Compute FIRST sets
#                         first_sets = parser.compute_first_sets()
                        
#                         # Compute FOLLOW sets
#                         follow_sets = parser.compute_follow_sets(first_sets)
                        
#                         # Store in session state
#                         st.session_state.parser = parser
#                         st.session_state.first_sets = first_sets
#                         st.session_state.follow_sets = follow_sets
#                     else:
#                         st.error("Failed to parse grammar. Please check your input.")
#                 else:
#                     st.error("Please enter grammar rules.")
        
#         with col2:
#             st.subheader("Analysis Results")
            
#             if hasattr(st.session_state, 'first_sets') and hasattr(st.session_state, 'follow_sets'):
#                 # FIRST sets
#                 st.write("**FIRST Sets:**")
#                 first_df = pd.DataFrame([
#                     {'Non-terminal': nt, 'FIRST': ', '.join(sorted(first_set))}
#                     for nt, first_set in st.session_state.first_sets.items()
#                 ])
#                 st.dataframe(first_df, use_container_width=True)
                
#                 # FOLLOW sets
#                 st.write("**FOLLOW Sets:**")
#                 follow_df = pd.DataFrame([
#                     {'Non-terminal': nt, 'FOLLOW': ', '.join(sorted(follow_set))}
#                     for nt, follow_set in st.session_state.follow_sets.items()
#                 ])
#                 st.dataframe(follow_df, use_container_width=True)
                
#                 # Grammar info
#                 with st.expander("Grammar Information"):
#                     parser = st.session_state.parser
#                     st.write(f"**Start Symbol:** {parser.start_symbol}")
#                     st.write(f"**Non-terminals:** {', '.join(sorted(parser.non_terminals))}")
#                     st.write(f"**Terminals:** {', '.join(sorted(parser.terminals))}")
                    
#                     st.write("**Production Rules:**")
#                     for nt, productions in parser.rules.items():
#                         for prod in productions:
#                             st.write(f"• {nt} → {' '.join(prod) if prod else 'ε'}")
    
#     with tab2:
#         st.header("Arithmetic Expression Parser & Evaluator")
        
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             st.subheader("Expression Input")
            
#             # Sample expression button
#             if st.button("Load Sample Expression", key="sample_expr"):
#                 st.session_state.expression_text = create_sample_expression()
            
#             expression = st.text_input(
#                 "Enter arithmetic expression:",
#                 value=st.session_state.get('expression_text', ''),
#                 help="Supports +, -, *, /, parentheses, numbers, and variables"
#             )
            
#             if st.button("Parse & Evaluate", key="parse_eval"):
#                 if expression.strip():
#                     expr_parser = ExpressionParser()
                    
#                     try:
#                         # Parse using recursive descent
#                         parse_tree, tree_dot = expr_parser.recursive_descent_parse(expression)
                        
#                         # Evaluate expression
#                         result = expr_parser.evaluate_expression(expression)
                        
#                         # Store results
#                         st.session_state.parse_tree_dot = tree_dot
#                         st.session_state.expression_result = result
#                         st.session_state.parsed_successfully = True
                        
#                         st.success("Expression parsed successfully!")
                        
#                     except Exception as e:
#                         st.error(f"Parse error: {str(e)}")
#                         st.session_state.parsed_successfully = False
#                 else:
#                     st.error("Please enter an expression.")
        
#         with col2:
#             st.subheader("Parse Results")
            
#             if st.session_state.get('parsed_successfully', False):
#                 # Show evaluation result
#                 if st.session_state.get('expression_result') is not None:
#                     st.metric("Evaluation Result", f"{st.session_state.expression_result}")
                
#                 # Show parse tree
#                 if st.session_state.get('parse_tree_dot'):
#                     st.write("**Parse Tree (Recursive Descent):**")
#                     try:
#                         st.graphviz_chart(st.session_state.parse_tree_dot)
#                     except Exception as e:
#                         st.error(f"Error displaying parse tree: {str(e)}")
#                         st.code(st.session_state.parse_tree_dot, language='dot')
    
#     # Help section
#     with st.expander("ℹ️ Help & Examples"):
#         st.markdown("""
#         ### Grammar Visualizer
        
#         **Grammar Format:**
#         ```
#         S -> A B | C
#         A -> a | epsilon
#         B -> b B | b
#         C -> c
#         ```
        
#         **Sample Grammars:**
        
#         1. **Arithmetic Expressions:**
#         ```
#         E -> E + T | T
#         T -> T * F | F
#         F -> ( E ) | id
#         ```
        
#         2. **Simple Language:**
#         ```
#         S -> if E then S else S | id = E
#         E -> E + T | T
#         T -> id | num
#         ```
        
#         ### Expression Evaluator
        
#         **Supported Operations:**
#         - Addition: `+`
#         - Subtraction: `-`
#         - Multiplication: `*`
#         - Division: `/`
#         - Parentheses: `(` `)`
#         - Numbers: `123`, `3.14`
#         - Variables: `x`, `y`, `var`
        
#         **Sample Expressions:**
#         - `3 + 4 * 5`
#         - `(2 + 3) * (4 - 1)`
#         - `x + y * 2`
#         """)

# if __name__ == "__main__":
#     main()






import streamlit as st
import pandas as pd
import re
import graphviz
import networkx as nx
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Grammar Visualizer & Expression Evaluator",
    page_icon="📝",
    layout="wide"
)

class GrammarParser:
    """Handles context-free grammar parsing and analysis"""
    
    def __init__(self):
        self.rules = defaultdict(list)
        self.non_terminals = set()
        self.terminals = set()
        self.start_symbol = None
        
    def parse_grammar(self, grammar_text: str) -> bool:
        """Parse grammar rules from text input"""
        try:
            self.rules.clear()
            self.non_terminals.clear()
            self.terminals.clear()
            
            lines = [line.strip() for line in grammar_text.strip().split('\n') if line.strip()]
            
            if not lines:
                return False
                
            self.start_symbol = lines[0].split('->')[0].strip()
            
            # First pass: identify all non-terminals
            for line in lines:
                if '->' not in line:
                    continue
                    
                left, right = line.split('->', 1)
                left = left.strip()
                
                if not left:
                    continue
                    
                self.non_terminals.add(left)
            
            # Second pass: parse productions and identify terminals
            for line in lines:
                if '->' not in line:
                    continue
                    
                left, right = line.split('->', 1)
                left = left.strip()
                
                if not left:
                    continue
                
                # Split productions by |
                productions = [p.strip() for p in right.split('|') if p.strip()]
                
                for prod in productions:
                    if prod.lower() in ['epsilon', 'ε']:
                        symbols = ['ε']
                    else:
                        symbols = prod.split()
                    
                    self.rules[left].append(symbols)
                    
                    # Identify terminals (symbols not in non_terminals)
                    for symbol in symbols:
                        if symbol not in ['epsilon', 'ε'] and symbol not in self.non_terminals:
                            self.terminals.add(symbol)
            
            return True
            
        except Exception as e:
            st.error(f"Error parsing grammar: {str(e)}")
            return False
    
    def compute_first_sets(self) -> Dict[str, Set[str]]:
        """Compute FIRST sets for all non-terminals"""
        first = defaultdict(set)
        
        # Initialize FIRST sets for terminals
        for terminal in self.terminals:
            first[terminal] = {terminal}
            
        # Initialize FIRST sets for non-terminals
        for nt in self.non_terminals:
            first[nt] = set()
            
        changed = True
        iterations = 0
        max_iterations = 50  # Prevent infinite loops
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for nt in self.non_terminals:
                old_size = len(first[nt])
                
                for production in self.rules[nt]:
                    if not production or production == ['epsilon'] or production == ['ε']:
                        first[nt].add('ε')
                        continue
                    
                    # Process each symbol in the production
                    all_have_epsilon = True
                    for i, symbol in enumerate(production):
                        if symbol in self.terminals:
                            # Terminal symbol - add it to FIRST set
                            first[nt].add(symbol)
                            all_have_epsilon = False
                            break
                        elif symbol in self.non_terminals:
                            # Non-terminal symbol - add its FIRST set (minus epsilon)
                            symbol_first = first[symbol] - {'ε'}
                            first[nt].update(symbol_first)
                            
                            # If this symbol doesn't have epsilon, we can't continue
                            if 'ε' not in first[symbol]:
                                all_have_epsilon = False
                                break
                        else:
                            # Unknown symbol - treat as terminal
                            first[nt].add(symbol)
                            all_have_epsilon = False
                            break
                    
                    # If all symbols in production can derive epsilon, add epsilon
                    if all_have_epsilon:
                        first[nt].add('ε')
                
                if len(first[nt]) > old_size:
                    changed = True
        
        # Return only non-terminal FIRST sets (terminals already have their own symbols)
        return {nt: first[nt] for nt in self.non_terminals}
    
    def compute_follow_sets(self, first_sets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Compute FOLLOW sets for all non-terminals"""
        follow = defaultdict(set)
        
        # Initialize
        for nt in self.non_terminals:
            follow[nt] = set()
            
        # Start symbol gets $
        if self.start_symbol:
            follow[self.start_symbol].add('$')
            
        changed = True
        while changed:
            changed = False
            
            for nt in self.non_terminals:
                for production in self.rules[nt]:
                    for i, symbol in enumerate(production):
                        if symbol in self.non_terminals:
                            old_size = len(follow[symbol])
                            
                            # Look at symbols after current symbol
                            beta = production[i + 1:]
                            
                            if not beta:  # Symbol is at end
                                follow[symbol].update(follow[nt])
                            else:
                                # Compute FIRST(beta)
                                first_beta = set()
                                for j, beta_symbol in enumerate(beta):
                                    if beta_symbol in self.terminals:
                                        first_beta.add(beta_symbol)
                                        break
                                    elif beta_symbol in self.non_terminals:
                                        first_beta.update(first_sets[beta_symbol] - {'ε'})
                                        if 'ε' not in first_sets[beta_symbol]:
                                            break
                                    
                                    # If all symbols in beta can derive epsilon
                                    if j == len(beta) - 1:
                                        first_beta.add('ε')
                                
                                follow[symbol].update(first_beta - {'ε'})
                                
                                if 'ε' in first_beta:
                                    follow[symbol].update(follow[nt])
                            
                            if len(follow[symbol]) > old_size:
                                changed = True
                                
        return dict(follow)

class ExpressionParser:
    """Handles arithmetic expression parsing"""
    
    def __init__(self):
        self.tokens = []
        self.pos = 0
        
    def tokenize(self, expression: str) -> List[Tuple[str, str]]:
        """Tokenize arithmetic expression"""
        tokens = []
        i = 0
        
        while i < len(expression):
            if expression[i].isspace():
                i += 1
                continue
                
            if expression[i].isdigit():
                num = ''
                while i < len(expression) and (expression[i].isdigit() or expression[i] == '.'):
                    num += expression[i]
                    i += 1
                tokens.append(('NUMBER', num))
            elif expression[i].isalpha():
                var = ''
                while i < len(expression) and expression[i].isalnum():
                    var += expression[i]
                    i += 1
                tokens.append(('ID', var))
            elif expression[i] in '+-*/()':
                tokens.append(('OP', expression[i]))
                i += 1
            else:
                raise ValueError(f"Invalid character: {expression[i]}")
                
        tokens.append(('EOF', ''))
        return tokens
    
    def recursive_descent_parse(self, expression: str) -> Tuple[any, str]:
        """Parse using recursive descent (top-down)"""
        try:
            self.tokens = self.tokenize(expression)
            self.pos = 0
            
            result = self.parse_expression()
            
            if self.current_token()[0] != 'EOF':
                raise ValueError("Unexpected tokens at end of expression")
                
            # Create parse tree visualization
            tree_dot = self.create_parse_tree_dot(result)
            
            return result, tree_dot
            
        except Exception as e:
            raise ValueError(f"Parse error: {str(e)}")
    
    def current_token(self):
        """Get current token"""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ('EOF', '')
    
    def consume_token(self, expected_type=None):
        """Consume current token"""
        token = self.current_token()
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token[0]}")
        self.pos += 1
        return token
    
    def parse_expression(self):
        """Parse expression: term (('+' | '-') term)*"""
        node = {'type': 'expression', 'children': []}
        left = self.parse_term()
        node['children'].append(left)
        
        while self.current_token()[1] in ['+', '-']:
            op = self.consume_token('OP')
            right = self.parse_term()
            node['children'].extend([{'type': 'operator', 'value': op[1]}, right])
            
        return node
    
    def parse_term(self):
        """Parse term: factor (('*' | '/') factor)*"""
        node = {'type': 'term', 'children': []}
        left = self.parse_factor()
        node['children'].append(left)
        
        while self.current_token()[1] in ['*', '/']:
            op = self.consume_token('OP')
            right = self.parse_factor()
            node['children'].extend([{'type': 'operator', 'value': op[1]}, right])
            
        return node
    
    def parse_factor(self):
        """Parse factor: number | id | '(' expression ')'"""
        token = self.current_token()
        
        if token[0] == 'NUMBER':
            self.consume_token('NUMBER')
            return {'type': 'number', 'value': token[1]}
        elif token[0] == 'ID':
            self.consume_token('ID')
            return {'type': 'id', 'value': token[1]}
        elif token[1] == '(':
            self.consume_token('OP')
            expr = self.parse_expression()
            self.consume_token('OP')  # closing )
            return {'type': 'parentheses', 'children': [expr]}
        else:
            raise ValueError(f"Unexpected token: {token}")
    
    def create_parse_tree_dot(self, node, counter=[0]) -> str:
        """Create DOT representation of parse tree"""
        dot = graphviz.Digraph()
        dot.attr(rankdir='TB')
        
        def add_node(n, parent_id=None):
            node_id = f"node_{counter[0]}"
            counter[0] += 1
            
            if n['type'] == 'number':
                dot.node(node_id, f"NUMBER\\n{n['value']}", shape='box', style='filled', fillcolor='lightblue')
            elif n['type'] == 'id':
                dot.node(node_id, f"ID\\n{n['value']}", shape='box', style='filled', fillcolor='lightgreen')
            elif n['type'] == 'operator':
                dot.node(node_id, f"OP\\n{n['value']}", shape='diamond', style='filled', fillcolor='orange')
            else:
                dot.node(node_id, n['type'].upper(), shape='ellipse', style='filled', fillcolor='lightgray')
            
            if parent_id:
                dot.edge(parent_id, node_id)
            
            if 'children' in n:
                for child in n['children']:
                    add_node(child, node_id)
                    
            return node_id
        
        add_node(node)
        return dot.source
    
    def evaluate_expression(self, expression: str) -> float:
        """Evaluate arithmetic expression"""
        try:
            # Simple evaluation using Python's eval (for demonstration)
            # In production, you'd want a safer evaluator
            safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            return eval(safe_expr)
        except:
            return None

def create_sample_grammar():
    """Create sample grammar for testing"""
    return """E -> E + T | T
T -> T * F | F
F -> ( E ) | id"""

def get_sample_grammars():
    """Get dictionary of sample grammars"""
    return {
        "1. Basic Arithmetic": """E -> E + T | T
T -> T * F | F
F -> ( E ) | id""",
        
        "2. If-Then-Else": """S -> if E then S else S | if E then S | id = E
E -> E + T | E - T | T
T -> T * F | T / F | F
F -> ( E ) | id | num""",
        
        "3. Simple Programming Language": """P -> D ; P | S ; P | epsilon
D -> int L | float L
L -> id , L | id
S -> id = E | if ( E ) S | while ( E ) S | { P }
E -> E + T | E - T | T
T -> T * F | T / F | F
F -> ( E ) | id | num""",
        
        "4. Balanced Parentheses": """S -> ( S ) S | epsilon""",
        
        "5. Simple Expression with Unary": """E -> E + T | E - T | T
T -> T * F | T / F | F
F -> + F | - F | ( E ) | id | num""",
        
        "6. List Structure": """L -> [ E ] | [ ]
E -> E , T | T
T -> id | num | L"""
    }

def create_sample_expression():
    """Create sample expression for testing"""
    return "3 + 4 * (5 - 2)"

def get_sample_expressions():
    """Get list of sample expressions"""
    return [
        "3 + 4 * (5 - 2)",
        "((10 + 5) * 2) / (7 - 2)",
        "a + b * c - d / e",
        "(x + y) * (z - w) + 15",
        "100 / (2 + 3) * 4 - 7"
    ]

def main():
    st.title("🔤 Interactive Grammar Visualizer & Expression Evaluator")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📝 Grammar Visualizer", "🧮 Expression Evaluator"])
    
    with tab1:
        st.header("Context-Free Grammar Analyzer")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Grammar Input")
            
            # Sample grammar selection
            st.write("**Sample Grammars:**")
            sample_grammars = get_sample_grammars()
            selected_grammar = st.selectbox(
                "Choose a sample grammar:",
                options=list(sample_grammars.keys()),
                index=0,
                key="grammar_selector"
            )
            
            if st.button("Load Selected Grammar", key="load_grammar"):
                st.session_state.grammar_text = sample_grammars[selected_grammar]
            
            grammar_text = st.text_area(
                "Enter grammar rules (one per line):",
                value=st.session_state.get('grammar_text', ''),
                height=200,
                help="Format: A -> B C | D\nUse 'epsilon' or 'ε' for empty production"
            )
            
            if st.button("Analyze Grammar", key="analyze"):
                if grammar_text.strip():
                    parser = GrammarParser()
                    
                    if parser.parse_grammar(grammar_text):
                        st.success("Grammar parsed successfully!")
                        
                        # Compute FIRST sets
                        first_sets = parser.compute_first_sets()
                        
                        # Compute FOLLOW sets
                        follow_sets = parser.compute_follow_sets(first_sets)
                        
                        # Store in session state
                        st.session_state.parser = parser
                        st.session_state.first_sets = first_sets
                        st.session_state.follow_sets = follow_sets
                    else:
                        st.error("Failed to parse grammar. Please check your input.")
                else:
                    st.error("Please enter grammar rules.")
        
        with col2:
            st.subheader("Analysis Results")
            
            if hasattr(st.session_state, 'first_sets') and hasattr(st.session_state, 'follow_sets'):
                # FIRST sets
                st.write("**FIRST Sets:**")
                first_df = pd.DataFrame([
                    {'Non-terminal': nt, 'FIRST': ', '.join(sorted(first_set))}
                    for nt, first_set in st.session_state.first_sets.items()
                ])
                st.dataframe(first_df, use_container_width=True)
                
                # FOLLOW sets
                st.write("**FOLLOW Sets:**")
                follow_df = pd.DataFrame([
                    {'Non-terminal': nt, 'FOLLOW': ', '.join(sorted(follow_set))}
                    for nt, follow_set in st.session_state.follow_sets.items()
                ])
                st.dataframe(follow_df, use_container_width=True)
                
                # Grammar info
                with st.expander("Grammar Information"):
                    parser = st.session_state.parser
                    st.write(f"**Start Symbol:** {parser.start_symbol}")
                    st.write(f"**Non-terminals:** {', '.join(sorted(parser.non_terminals))}")
                    st.write(f"**Terminals:** {', '.join(sorted(parser.terminals))}")
                    
                    st.write("**Production Rules:**")
                    for nt, productions in parser.rules.items():
                        for prod in productions:
                            st.write(f"• {nt} → {' '.join(prod) if prod else 'ε'}")
                
                # Debug information for FIRST sets
                with st.expander("🔍 FIRST Set Computation Details"):
                    st.write("**How FIRST sets are computed:**")
                    st.write("1. For each terminal symbol: FIRST(terminal) = {terminal}")
                    st.write("2. For each non-terminal A:")
                    st.write("   - If A → ε, then add ε to FIRST(A)")
                    st.write("   - If A → X₁X₂...Xₙ:")
                    st.write("     • Add FIRST(X₁) - {ε} to FIRST(A)")
                    st.write("     • If ε ∈ FIRST(X₁), add FIRST(X₂) - {ε} to FIRST(A)")
                    st.write("     • Continue until Xᵢ where ε ∉ FIRST(Xᵢ) or all symbols processed")
                    st.write("     • If ε ∈ FIRST(Xᵢ) for all i, then add ε to FIRST(A)")
                    
                    st.write("**Terminal FIRST sets:**")
                    for terminal in sorted(parser.terminals):
                        if terminal != 'ε':
                            st.write(f"• FIRST({terminal}) = {{{terminal}}}")
                            
                    st.write("**Non-terminal productions analysis:**")
                    for nt in sorted(parser.non_terminals):
                        st.write(f"**{nt} productions:**")
                        for i, prod in enumerate(parser.rules[nt]):
                            if prod == ['ε']:
                                st.write(f"  {i+1}. {nt} → ε  ⟹  adds ε to FIRST({nt})")
                            else:
                                analysis = f"  {i+1}. {nt} → {' '.join(prod)}  ⟹  "
                                first_symbol = prod[0]
                                if first_symbol in parser.terminals:
                                    analysis += f"adds {{{first_symbol}}} to FIRST({nt})"
                                else:
                                    analysis += f"adds FIRST({first_symbol}) to FIRST({nt})"
                                st.write(analysis)
    
    with tab2:
        st.header("Arithmetic Expression Parser & Evaluator")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Expression Input")
            
            # Sample expression selection
            st.write("**Sample Expressions:**")
            sample_expressions = get_sample_expressions()
            selected_expression = st.selectbox(
                "Choose a sample expression:",
                options=[f"{i+1}. {expr}" for i, expr in enumerate(sample_expressions)],
                index=0,
                key="expression_selector"
            )
            
            if st.button("Load Selected Expression", key="load_expression"):
                # Extract just the expression part (after the number and dot)
                expr_text = selected_expression.split(". ", 1)[1]
                st.session_state.expression_text = expr_text
            
            expression = st.text_input(
                "Enter arithmetic expression:",
                value=st.session_state.get('expression_text', ''),
                help="Supports +, -, *, /, parentheses, numbers, and variables"
            )
            
            if st.button("Parse & Evaluate", key="parse_eval"):
                if expression.strip():
                    expr_parser = ExpressionParser()
                    
                    try:
                        # Parse using recursive descent
                        parse_tree, tree_dot = expr_parser.recursive_descent_parse(expression)
                        
                        # Evaluate expression
                        result = expr_parser.evaluate_expression(expression)
                        
                        # Store results
                        st.session_state.parse_tree_dot = tree_dot
                        st.session_state.expression_result = result
                        st.session_state.parsed_successfully = True
                        
                        st.success("Expression parsed successfully!")
                        
                    except Exception as e:
                        st.error(f"Parse error: {str(e)}")
                        st.session_state.parsed_successfully = False
                else:
                    st.error("Please enter an expression.")
        
        with col2:
            st.subheader("Parse Results")
            
            if st.session_state.get('parsed_successfully', False):
                # Show evaluation result
                if st.session_state.get('expression_result') is not None:
                    st.metric("Evaluation Result", f"{st.session_state.expression_result}")
                
                # Show parse tree
                if st.session_state.get('parse_tree_dot'):
                    st.write("**Parse Tree (Recursive Descent):**")
                    try:
                        st.graphviz_chart(st.session_state.parse_tree_dot)
                    except Exception as e:
                        st.error(f"Error displaying parse tree: {str(e)}")
                        st.code(st.session_state.parse_tree_dot, language='dot')
    
    # Help section
    with st.expander("ℹ️ Help & Examples"):
        st.markdown("""
        ### Grammar Visualizer
        
        **Grammar Format:**
        ```
        S -> A B | C
        A -> a | epsilon
        B -> b B | b
        C -> c
        ```
        
        **Sample Grammars:**
        
        1. **Basic Arithmetic:**
        ```
        E -> E + T | T
        T -> T * F | F
        F -> ( E ) | id
        ```
        
        2. **If-Then-Else Statements:**
        ```
        S -> if E then S else S | if E then S | id = E
        E -> E + T | E - T | T
        T -> T * F | T / F | F
        F -> ( E ) | id | num
        ```
        
        3. **Simple Programming Language:**
        ```
        P -> D ; P | S ; P | epsilon
        D -> int L | float L
        L -> id , L | id
        S -> id = E | if ( E ) S | while ( E ) S | { P }
        E -> E + T | E - T | T
        T -> T * F | T / F | F
        F -> ( E ) | id | num
        ```
        
        4. **Balanced Parentheses:**
        ```
        S -> ( S ) S | epsilon
        ```
        
        5. **Expression with Unary Operators:**
        ```
        E -> E + T | E - T | T
        T -> T * F | T / F | F
        F -> + F | - F | ( E ) | id | num
        ```
        
        6. **List Structure:**
        ```
        L -> [ E ] | [ ]
        E -> E , T | T
        T -> id | num | L
        ```
        
        ### Expression Evaluator
        
        **Supported Operations:**
        - Addition: `+`
        - Subtraction: `-`
        - Multiplication: `*`
        - Division: `/`
        - Parentheses: `(` `)`
        - Numbers: `123`, `3.14`
        - Variables: `x`, `y`, `var`
        
        **Sample Expressions:**
        - `3 + 4 * (5 - 2)` - Basic arithmetic with precedence
        - `((10 + 5) * 2) / (7 - 2)` - Nested parentheses and division
        - `a + b * c - d / e` - Variables with mixed operations
        - `(x + y) * (z - w) + 15` - Complex variable expressions
        - `100 / (2 + 3) * 4 - 7` - Order of operations test
        """)

if __name__ == "__main__":
    main()