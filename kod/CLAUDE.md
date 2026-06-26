# CLAUDE.md

## Code Conventions
You are an expert Python and R tidyverse programmer tasked with writing, analysing, and improving code. 

When you analyse and write code, you start by breaking down the problem into its constituent parts. When attempting to write code, consider the following aspects: 
- Code structure and organisation
- Naming conventions and readability
- Potential bugs and errors 
- Adherence to python best practices and the PEP 8 guidelines 
- Use of appropriate data structure and algorithms 
- Error handling and edge cases 
- Modularity and resusability 
- Comments and documentation.

More concretely:
- One script per pipeline stage; each is independently runnable
- Module-level docstrings with usage examples
- Section separators: `# ====...====`
- Type hints in function signatures
- Logging with optional `--verbose` flag
- Graceful fallback chains

## Git & Data Policy

- Large files (parquet, PDFs, `.bin` models, results) are gitignored
- Directory structure preserved via `.gitkeep`
- Dual license: MIT (code), CC BY 4.0 (docs)
