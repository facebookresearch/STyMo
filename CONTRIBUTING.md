# Contributing to STyMo

We want to make contributing to this project as easy and transparent as
possible.

## Getting Started

STyMo is a research codebase for few-shot motion style transfer. It is written
in Python with PyTorch and uses [pymotion](https://github.com/UPC-ViRVIG/pymotion)
for BVH I/O and skeleton operations, and Blender for visualization.

### Installation for development

1. **Fork and clone:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/STyMo.git
   cd STyMo
   git remote add upstream https://github.com/facebookresearch/STyMo.git
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate   # macOS/Linux
   # or .\env\Scripts\activate   # Windows
   ```

3. **Install PyTorch:** follow https://pytorch.org/get-started/locally/ for
   your CUDA configuration.

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install Blender:** https://www.blender.org/ (required for the GUI app and
   visualization).

## Development Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes. Please follow existing code style:
   - Python: follow PEP 8, use type hints where practical, add docstrings for
     public functions
   - Keep changes focused and well-scoped

3. Test your changes locally:
   ```bash
   # Train on example data
   python src/stylizer_train.py motion/mocha/angry motion/mocha/neutral/Dance_Neutral_AverageJoe.bvh --epochs 10 --name test
   # Run inference GUI
   python src/stylizer_app.py motion/mocha/angry/results/Transformer-test motion/mocha/neutral/Dance_Neutral_AverageJoe.bvh
   ```

4. Commit and push:
   ```bash
   git commit -m "Clear description of change"
   git push origin feature/your-feature-name
   ```

5. Open a Pull Request on GitHub.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests or example usage.
3. If you've changed APIs, update the documentation and USAGE.md.
4. Ensure existing functionality still works on the example motions.
5. Make sure Python files include the standard Meta copyright header:
   ```
   # Copyright (c) Meta Platforms, Inc. and affiliates.
   #
   # This source code is licensed under the MIT license found in the
   # LICENSE file in the root directory of this source tree.
   ```
6. If you haven't already, complete the Contributor License Agreement (CLA).

## Reporting Issues

Use GitHub Issues to report bugs or request features. Please include:
- Python version, PyTorch version, OS
- Command or steps to reproduce
- Relevant BVH files or error logs if applicable

## License

By contributing to STyMo, you agree that your contributions will be licensed
under the MIT License found in the LICENSE file in the root directory of this
source tree.
