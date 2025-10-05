# Interactive Grammar Visualizer & Expression Evaluator

A comprehensive Streamlit web application for educational purposes that combines context-free grammar analysis with arithmetic expression parsing and evaluation.



## Features

### 📝 Grammar Visualizer
- **Grammar Parsing**: Input context-free grammar rules in standard notation
- **FIRST Sets**: Automatically compute FIRST sets for all non-terminals
- **FOLLOW Sets**: Calculate FOLLOW sets based on grammar rules
- **Grammar Analysis**: Display terminals, non-terminals, and production rules
- **Input Validation**: Error handling for malformed grammar rules

### 🧮 Expression Evaluator
- **Expression Parsing**: Parse arithmetic expressions using recursive descent
- **Parse Tree Visualization**: Generate visual parse trees using Graphviz
- **Expression Evaluation**: Calculate numerical results
- **Error Handling**: Comprehensive error messages for invalid expressions
- **Token Analysis**: Break down expressions into tokens

## Installation

### Prerequisites
- Python 3.10 or higher
- Graphviz system package (for visualization)

### Install Graphviz System Package

**On Ubuntu/Debian:**
```bash
sudo apt-get install graphviz
```

**On macOS:**
```bash
brew install graphviz
```

**On Windows:**
1. Download Graphviz from https://graphviz.org/download/
2. Install and add to PATH

### Install Python Dependencies

1. Clone or download the project files
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. Navigate to the project directory
2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open your web browser and go to `http://localhost:8501`

### Using the Grammar Visualizer

1. **Input Grammar**: Enter context-free grammar rules in the text area
   - Format: `A -> B C | D`
   - Use `epsilon` or `ε` for empty productions
   - One rule per line

2. **Sample Grammar**: Click "Load Sample Grammar" for a pre-defined example

3. **Analyze**: Click "Analyze Grammar" to compute FIRST and FOLLOW sets

4. **View Results**: 
   - FIRST and FOLLOW sets displayed in tables
   - Grammar information in expandable section

### Using the Expression Evaluator

1. **Input Expression**: Enter arithmetic expressions
   - Supports: `+`, `-`, `*`, `/`, `()`, numbers, variables
   - Example: `3 + 4 * (5 - 2)`

2. **Sample Expression**: Click "Load Sample Expression" for a pre-defined example

3. **Parse & Evaluate**: Click to parse and evaluate the expression

4. **View Results**:
   - Numerical evaluation result
   - Visual parse tree diagram

## Grammar Format Examples

### Basic Arithmetic Grammar
```
E -> E + T | T
T -> T * F | F
F -> ( E ) | id
```

### Simple Programming Language
```
S -> if E then S else S | id = E
E -> E + T | T
T -> id | num
```

### Recursive Grammar
```
S -> a S b | epsilon
```

## Expression Format Examples

- Simple: `3 + 4`
- Complex: `(2 + 3) * (4 - 1) / 2`
- With variables: `x + y * z`
- Nested: `((1 + 2) * 3) + 4`

## Technical Implementation

### Grammar Analysis
- **FIRST Sets**: Computed using iterative algorithm
- **FOLLOW Sets**: Based on FIRST sets and grammar structure
- **Validation**: Checks for proper grammar syntax

### Expression Parsing
- **Tokenization**: Converts input to tokens (numbers, operators, identifiers)
- **Recursive Descent**: Top-down parsing approach
- **Parse Tree**: Visual representation using Graphviz
- **Evaluation**: Safe expression evaluation

### Error Handling
- Grammar parsing errors with specific messages
- Expression parsing errors with context
- Input validation and user-friendly feedback

## Project Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and display
- **graphviz**: Graph visualization library
- **networkx**: Network analysis library
- **matplotlib**: Plotting library

## Troubleshooting

### Common Issues

1. **Graphviz not found**: Install system Graphviz package
2. **Parse tree not displaying**: Check Graphviz installation
3. **Grammar parsing errors**: Verify grammar format and syntax

### Error Messages

- "Failed to parse grammar": Check rule syntax and format
- "Parse error": Verify expression syntax
- "Invalid character": Remove unsupported characters from expression

## Educational Value

This application helps students understand:

- Context-free grammar concepts
- FIRST and FOLLOW set computation
- Parse tree construction
- Top-down parsing techniques
- Expression evaluation
- Compiler design principles

## Future Enhancements

Potential improvements:
- Bottom-up parsing implementation
- LL(1) and SLR parse table generation
- More sophisticated error recovery
- Interactive parse step visualization
- Grammar ambiguity detection
- Extended expression support (functions, arrays)

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## License

This project is for educational purposes. Feel free to use and modify as needed.
