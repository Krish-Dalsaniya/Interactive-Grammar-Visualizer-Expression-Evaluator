import streamlit as st
import pandas as pd
from collections import defaultdict
import graphviz

# -----------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------
st.set_page_config(
    page_title="Interactive Grammar Visualizer",
    page_icon="🧩",
    layout="wide"
)

# -----------------------------------------------
# Grammar Parser Class
# -----------------------------------------------
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

            # Identify all non-terminals
            for line in lines:
                if '->' in line:
                    left = line.split('->')[0].strip()
                    self.non_terminals.add(left)

            # Parse productions and identify terminals
            for line in lines:
                if '->' not in line:
                    continue
                left, right = line.split('->', 1)
                left = left.strip()
                productions = [p.strip() for p in right.split('|') if p.strip()]

                for prod in productions:
                    if prod.lower() in ['epsilon', 'ε']:
                        symbols = ['ε']
                    else:
                        symbols = prod.split()

                    self.rules[left].append(symbols)

                    for symbol in symbols:
                        if symbol not in ['ε'] and symbol not in self.non_terminals:
                            self.terminals.add(symbol)
            return True
        except Exception as e:
            st.error(f"Error parsing grammar: {str(e)}")
            return False

    def compute_first_sets(self):
        """Compute FIRST sets for all non-terminals"""
        first = defaultdict(set)

        for terminal in self.terminals:
            first[terminal] = {terminal}

        for nt in self.non_terminals:
            first[nt] = set()

        changed = True
        while changed:
            changed = False
            for nt in self.non_terminals:
                old_size = len(first[nt])
                for production in self.rules[nt]:
                    if production == ['ε']:
                        first[nt].add('ε')
                        continue

                    all_have_epsilon = True
                    for symbol in production:
                        if symbol in self.terminals:
                            first[nt].add(symbol)
                            all_have_epsilon = False
                            break
                        elif symbol in self.non_terminals:
                            first[nt].update(first[symbol] - {'ε'})
                            if 'ε' not in first[symbol]:
                                all_have_epsilon = False
                                break
                        else:
                            first[nt].add(symbol)
                            all_have_epsilon = False
                            break
                    if all_have_epsilon:
                        first[nt].add('ε')

                if len(first[nt]) > old_size:
                    changed = True

        return {nt: first[nt] for nt in self.non_terminals}

    def compute_follow_sets(self, first_sets):
        """Compute FOLLOW sets for all non-terminals"""
        follow = defaultdict(set)
        for nt in self.non_terminals:
            follow[nt] = set()

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
                            beta = production[i + 1:]
                            if not beta:
                                follow[symbol].update(follow[nt])
                            else:
                                first_beta = set()
                                for j, b in enumerate(beta):
                                    if b in self.terminals:
                                        first_beta.add(b)
                                        break
                                    elif b in self.non_terminals:
                                        first_beta.update(first_sets[b] - {'ε'})
                                        if 'ε' not in first_sets[b]:
                                            break
                                    if j == len(beta) - 1:
                                        first_beta.add('ε')

                                follow[symbol].update(first_beta - {'ε'})
                                if 'ε' in first_beta:
                                    follow[symbol].update(follow[nt])

                            if len(follow[symbol]) > old_size:
                                changed = True

        return dict(follow)


# -----------------------------------------------
# Sample Grammar
# -----------------------------------------------
def get_sample_grammars():
    return {
        "Basic Arithmetic": """E -> E + T | T
T -> T * F | F
F -> ( E ) | id""",
        "If-Then-Else": """S -> if E then S else S | if E then S | id = E
E -> E + T | E - T | T
T -> T * F | T / F | F
F -> ( E ) | id | num""",
        "Balanced Parentheses": """S -> ( S ) S | ε""",
        "Simple Programming": """P -> D ; P | S ; P | ε
D -> int L | float L
L -> id , L | id
S -> id = E | if ( E ) S | while ( E ) S | { P }
E -> E + T | E - T | T
T -> T * F | T / F | F
F -> ( E ) | id | num"""
    }

# -----------------------------------------------
# Streamlit App UI
# -----------------------------------------------
def main():
    st.title("🧩 Interactive Grammar Visualizer")
    st.markdown("---")

    st.header("Context-Free Grammar Analyzer")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Grammar Input")

        sample_grammars = get_sample_grammars()
        selected_grammar = st.selectbox(
            "Choose a sample grammar:",
            options=list(sample_grammars.keys()),
            index=0
        )

        if st.button("Load Selected Grammar"):
            st.session_state.grammar_text = sample_grammars[selected_grammar]

        grammar_text = st.text_area(
            "Enter grammar rules (one per line):",
            value=st.session_state.get('grammar_text', ''),
            height=200,
            help="Format: A -> B C | D\nUse 'ε' or 'epsilon' for empty production"
        )

        if st.button("Analyze Grammar"):
            if grammar_text.strip():
                parser = GrammarParser()
                if parser.parse_grammar(grammar_text):
                    st.success("Grammar parsed successfully!")

                    first_sets = parser.compute_first_sets()
                    follow_sets = parser.compute_follow_sets(first_sets)

                    st.session_state.parser = parser
                    st.session_state.first_sets = first_sets
                    st.session_state.follow_sets = follow_sets
                else:
                    st.error("Failed to parse grammar. Check your input.")
            else:
                st.error("Please enter grammar rules.")

    with col2:
        st.subheader("Analysis Results")

        if hasattr(st.session_state, 'first_sets') and hasattr(st.session_state, 'follow_sets'):
            st.write("### FIRST Sets:")
            first_df = pd.DataFrame([
                {'Non-terminal': nt, 'FIRST': ', '.join(sorted(fs))}
                for nt, fs in st.session_state.first_sets.items()
            ])
            st.dataframe(first_df, use_container_width=True)

            st.write("### FOLLOW Sets:")
            follow_df = pd.DataFrame([
                {'Non-terminal': nt, 'FOLLOW': ', '.join(sorted(fs))}
                for nt, fs in st.session_state.follow_sets.items()
            ])
            st.dataframe(follow_df, use_container_width=True)

            with st.expander("Grammar Details"):
                parser = st.session_state.parser
                st.write(f"**Start Symbol:** {parser.start_symbol}")
                st.write(f"**Non-terminals:** {', '.join(sorted(parser.non_terminals))}")
                st.write(f"**Terminals:** {', '.join(sorted(parser.terminals))}")
                st.write("**Productions:**")
                for nt, prods in parser.rules.items():
                    for prod in prods:
                        st.write(f"• {nt} → {' '.join(prod)}")

            with st.expander("📘 How FIRST & FOLLOW are Computed"):
                st.markdown("""
                **FIRST Set Rules:**
                - If X is a terminal, FIRST(X) = {X}
                - If X → ε, then add ε to FIRST(X)
                - If X → Y₁Y₂…Yn, add FIRST(Y₁) to FIRST(X)
                  - If ε ∈ FIRST(Y₁), then add FIRST(Y₂), and so on
                  - If all Yi derive ε, then add ε to FIRST(X)
                
                **FOLLOW Set Rules:**
                - For the start symbol, add `$` to FOLLOW(S)
                - If A → αBβ, add FIRST(β) - {ε} to FOLLOW(B)
                - If A → αB or A → αBβ and ε ∈ FIRST(β), add FOLLOW(A) to FOLLOW(B)
                """)

    st.markdown("---")
    st.info("💡 This tool helps visualize FIRST and FOLLOW sets in Context-Free Grammars — a key concept in Compiler Design.")


# -----------------------------------------------
# Run App
# -----------------------------------------------
if __name__ == "__main__":
    main()
