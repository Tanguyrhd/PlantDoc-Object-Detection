# Makefile for PlantDoc Object Detection Pipeline
# Author: Your name
# Description: Automate dataset preparation and environment setup

.PHONY: help install clean clean-binary clean-diseases clean-species clean-all run-binary run-diseases run-species run-all test

# Default target
help:
	@echo "PlantDoc Object Detection - Available commands:"
	@echo ""
	@echo "Environment Setup:"
	@echo "  make install          Install Python dependencies"
	@echo "  make install-dev      Install with development tools"
	@echo ""
	@echo "Clean Operations:"
	@echo "  make clean-binary     Remove binary classification dataset"
	@echo "  make clean-diseases   Remove disease classification dataset"
	@echo "  make clean-species    Remove species classification dataset"
	@echo "  make clean-all        Remove all generated datasets"
	@echo ""
	@echo "Pipeline Execution:"
	@echo "  make run-binary       Run binary classification pipeline"
	@echo "  make run-diseases     Run disease classification pipeline"
	@echo "  make run-species      Run species classification pipeline"
	@echo "  make run-all          Run all three pipelines sequentially"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run tests"
	@echo "  make check-env        Check if .env file is configured"
	@echo ""
	@echo "Complete Workflows:"
	@echo "  make fresh-start      Clean all + run all pipelines"
	@echo "  make setup            Install + check environment"

# ====== ENVIRONMENT SETUP ======

install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

install-dev: install
	@echo "📦 Installing development tools..."
	pip install jupyter notebook ipykernel
	@echo "✅ Development environment ready!"

# ====== CLEAN OPERATIONS ======

clean-binary:
	@echo "🧹 Cleaning binary classification dataset..."
	@if [ -d "dataset/binary" ]; then \
		rm -rf dataset/binary; \
		echo "✅ Removed dataset/binary"; \
	else \
		echo "ℹ️  dataset/binary does not exist"; \
	fi

clean-diseases:
	@echo "🧹 Cleaning disease classification dataset..."
	@if [ -d "dataset/diseases" ]; then \
		rm -rf dataset/diseases; \
		echo "✅ Removed dataset/diseases"; \
	else \
		echo "ℹ️  dataset/diseases does not exist"; \
	fi

clean-species:
	@echo "🧹 Cleaning species classification dataset..."
	@if [ -d "dataset/species" ]; then \
		rm -rf dataset/species; \
		echo "✅ Removed dataset/species"; \
	else \
		echo "ℹ️  dataset/species does not exist"; \
	fi

clean-all: clean-binary clean-diseases clean-species
	@echo "🧹 Cleaning all generated datasets..."
	@if [ -d "dataset/disease" ]; then \
		rm -rf dataset/disease; \
		echo "✅ Removed dataset/disease (old pipeline)"; \
	fi
	@echo "✅ All datasets cleaned!"

# ====== PIPELINE EXECUTION ======

run-binary: check-env
	@echo "🚀 Running binary classification pipeline..."
	@echo "⚠️  This will process the training and test data"
	jupyter nbconvert --to notebook --execute notebooks/pipeline_binary.ipynb --output pipeline_binary_executed.ipynb
	@echo "✅ Binary pipeline completed! Output: dataset/binary/"

run-diseases: check-env
	@echo "🚀 Running disease classification pipeline..."
	@echo "⚠️  This will process the training and test data"
	jupyter nbconvert --to notebook --execute notebooks/pipeline_diseases.ipynb --output pipeline_diseases_executed.ipynb
	@echo "✅ Disease pipeline completed! Output: dataset/diseases/"

run-species: check-env
	@echo "🚀 Running species classification pipeline..."
	@echo "⚠️  This will process the training and test data"
	jupyter nbconvert --to notebook --execute notebooks/pipeline_species.ipynb --output pipeline_species_executed.ipynb
	@echo "✅ Species pipeline completed! Output: dataset/species/"

run-all: check-env
	@echo "🚀 Running ALL pipelines..."
	@$(MAKE) run-binary
	@$(MAKE) run-diseases
	@$(MAKE) run-species
	@echo "✅ All pipelines completed!"

# ====== TESTING & VALIDATION ======

check-env:
	@echo "🔍 Checking environment configuration..."
	@if [ ! -f ".env" ]; then \
		echo "❌ Error: .env file not found!"; \
		echo "Please create a .env file with your configuration."; \
		exit 1; \
	fi
	@if [ ! -f "dataset/train_labels.csv" ]; then \
		echo "❌ Error: dataset/train_labels.csv not found!"; \
		echo "Please ensure your dataset is in the correct location."; \
		exit 1; \
	fi
	@echo "✅ Environment check passed!"

test:
	@echo "🧪 Running tests..."
	@if command -v pytest > /dev/null; then \
		pytest tests/ -v; \
	else \
		echo "⚠️  pytest not installed. Run 'make install-dev'"; \
	fi

# ====== COMPLETE WORKFLOWS ======

setup: install check-env
	@echo "✅ Setup complete! Ready to run pipelines."

fresh-start: clean-all run-all
	@echo "✅ Fresh start complete! All datasets regenerated."

# ====== INFO ======

info:
	@echo "📊 Dataset Information:"
	@echo ""
	@echo "Binary Classification Dataset:"
	@if [ -d "dataset/binary" ]; then \
		echo "  Status: ✅ Exists"; \
		echo "  Train images: $$(find dataset/binary/images/train -type f 2>/dev/null | wc -l)"; \
		echo "  Val images: $$(find dataset/binary/images/val -type f 2>/dev/null | wc -l)"; \
	else \
		echo "  Status: ❌ Not generated"; \
	fi
	@echo ""
	@echo "Disease Classification Dataset:"
	@if [ -d "dataset/diseases" ]; then \
		echo "  Status: ✅ Exists"; \
		echo "  Train images: $$(find dataset/diseases/images/train -type f 2>/dev/null | wc -l)"; \
		echo "  Val images: $$(find dataset/diseases/images/val -type f 2>/dev/null | wc -l)"; \
	else \
		echo "  Status: ❌ Not generated"; \
	fi
	@echo ""
	@echo "Species Classification Dataset:"
	@if [ -d "dataset/species" ]; then \
		echo "  Status: ✅ Exists"; \
		echo "  Train images: $$(find dataset/species/images/train -type f 2>/dev/null | wc -l)"; \
		echo "  Val images: $$(find dataset/species/images/val -type f 2>/dev/null | wc -l)"; \
	else \
		echo "  Status: ❌ Not generated"; \
	fi
