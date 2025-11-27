# Makefile for PlantDoc Object Detection Pipeline
# Automate dataset preparation and pipeline execution

.PHONY: help install clean clean-binary clean-diseases clean-species clean-all \
        run-binary run-diseases run-species run-all \
        test test-config test-binary test-species test-diseases \
        check-env setup fresh-start info

# Default target
help:
	@echo "PlantDoc Object Detection - Available commands:"
	@echo ""
	@echo "Environment Setup:"
	@echo "  make install          Install Python dependencies"
	@echo "  make check-env        Check if .env file is configured"
	@echo "  make setup            Install + check environment"
	@echo ""
	@echo "Clean Operations:"
	@echo "  make clean-binary     Remove binary classification dataset"
	@echo "  make clean-diseases   Remove disease classification dataset"
	@echo "  make clean-species    Remove species classification dataset"
	@echo "  make clean-all        Remove all generated datasets"
	@echo ""
	@echo "Pipeline Execution (Production):"
	@echo "  make run-binary       Run binary classification pipeline"
	@echo "  make run-diseases     Run disease classification pipeline"
	@echo "  make run-species      Run species classification pipeline"
	@echo "  make run-all          Run all three pipelines sequentially"
	@echo ""
	@echo "Information:"
	@echo "  make info             Show dataset information and stats"
	@echo ""
	@echo "Complete Workflows:"
	@echo "  make fresh-start      Clean all + run all pipelines"

# ====== ENVIRONMENT SETUP ======

install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

check-env:
	@echo "🔍 Checking environment configuration..."
	@if [ ! -f ".env" ]; then \
		echo "❌ Error: .env file not found!"; \
		echo "Please create a .env file with your configuration."; \
		exit 1; \
	fi
	@if [ ! -f "dataset/train_labels.csv" ] || [ ! -f "dataset/test_labels.csv" ]; then \
		echo "❌ Error: dataset csv files not found!"; \
		echo "Please ensure your dataset is in the correct location."; \
		exit 1; \
	fi
	@echo "✅ Environment check passed!"

setup: install check-env
	@echo "✅ Setup complete! Ready to run pipelines."

# ====== CLEAN OPERATIONS ======

clean-binary:
	@echo "Cleaning binary classification dataset..."
	@if [ -d "dataset/binary" ]; then \
		rm -rf dataset/binary; \
		rm -rf dataset/binary.zip; \
		echo "✅ Removed dataset/binary and zip file associated"; \
	else \
		echo "ℹ️  dataset/binary does not exist"; \
	fi

clean-diseases:
	@echo "Cleaning disease classification dataset..."
	@if [ -d "dataset/diseases" ]; then \
		rm -rf dataset/diseases; \
		rm -rf dataset/diseases.zip; \
		echo "✅ Removed dataset/diseases"; \
	else \
		echo "ℹ️  dataset/diseases does not exist"; \
	fi

clean-species:
	@echo "Cleaning species classification dataset..."
	@if [ -d "dataset/species" ]; then \
		rm -rf dataset/species; \
		rm -rf dataset/species.zip; \
		echo "✅ Removed dataset/species"; \
	else \
		echo "ℹ️  dataset/species does not exist"; \
	fi

clean-all: clean-binary clean-diseases clean-species
	@echo "✅ All datasets cleaned!"

# ====== PIPELINE EXECUTION (Production) ======

run-binary: check-env
	@echo "🚀 Running binary classification pipeline..."
	python -c "from src.utils.logging_config import setup_logging; from src.config import PipelineConfig; from src.pipelines import BinaryPipeline; pipeline = BinaryPipeline(PipelineConfig()); pipeline.run()"
	@echo "✅ Binary pipeline completed! Output: dataset/binary/"

run-species: check-env
	@echo "🚀 Running species classification pipeline..."
	python -c "from src.utils.logging_config import setup_logging; from src.config import PipelineConfig; from src.pipelines import SpeciesPipeline; pipeline = SpeciesPipeline(PipelineConfig()); pipeline.run()"
	@echo "✅ Species pipeline completed! Output: dataset/species/"

run-diseases: check-env
	@echo "🚀 Running disease classification pipeline..."
	python -c "from src.utils.logging_config import setup_logging; from src.config import PipelineConfig; from src.pipelines import DiseasePipeline; pipeline = DiseasePipeline(PipelineConfig()); pipeline.run()"
	@echo "✅ Disease pipeline completed! Output: dataset/diseases/"

run-all: check-env
	@echo "🚀 Running ALL pipelines..."
	@$(MAKE) run-binary
	@$(MAKE) run-species
	@$(MAKE) run-diseases
	@echo "✅ All pipelines completed!"

# ====== COMPLETE WORKFLOWS ======

fresh-start: clean-all run-all
	@echo "✅ Fresh start complete! All datasets regenerated."

# ====== INFORMATION ======

info:
	@echo "📊 Dataset Information:"
	@echo ""
	@echo "Binary Classification Dataset:"
	@if [ -d "dataset/binary" ]; then \
		echo "  Status: ✅ Exists"; \
		echo "  Train images: $$(find dataset/binary/images/train -type f 2>/dev/null | wc -l)"; \
		echo "  Val images: $$(find dataset/binary/images/val -type f 2>/dev/null | wc -l)"; \
		if [ -f "dataset/binary/dataset.yaml" ]; then \
			echo "  Config: ✅ dataset/binary/dataset.yaml"; \
		fi \
	else \
		echo "  Status: ❌ Not generated"; \
	fi
	@echo ""
	@echo "Species Classification Dataset:"
	@if [ -d "dataset/species" ]; then \
		echo "  Status: ✅ Exists"; \
		echo "  Train images: $$(find dataset/species/images/train -type f 2>/dev/null | wc -l)"; \
		echo "  Val images: $$(find dataset/species/images/val -type f 2>/dev/null | wc -l)"; \
		if [ -f "dataset/species/dataset.yaml" ]; then \
			echo "  Config: ✅ dataset/species/dataset.yaml"; \
		fi \
	else \
		echo "  Status: ❌ Not generated"; \
	fi
	@echo ""
	@echo "Disease Classification Dataset:"
	@if [ -d "dataset/diseases" ]; then \
		echo "  Status: ✅ Exists"; \
		echo "  Train images: $$(find dataset/diseases/images/train -type f 2>/dev/null | wc -l)"; \
		echo "  Val images: $$(find dataset/diseases/images/val -type f 2>/dev/null | wc -l)"; \
		if [ -f "dataset/diseases/dataset.yaml" ]; then \
			echo "  Config: ✅ dataset/diseases/dataset.yaml"; \
		fi \
	else \
		echo "  Status: ❌ Not generated"; \
	fi
	@echo ""

# ====== API & DOCKER ======

run_api:
	uvicorn api.fast:app --reload

	test_api_root:
	pytest \
	tests/api/test_endpoints.py::test_root_is_up --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_root_returns_greeting --asyncio-mode=strict -W "ignore"

test_api_on_docker:
	pytest \
	tests/api/test_docker_endpoints.py --asyncio-mode=strict -W "ignore"

test_api_on_prod:
	pytest \
	tests/api/test_cloud_endpoints.py --asyncio-mode=strict -W "ignore"

# Variables
IMAGE_NAME = plantdoc-api
CONTAINER_NAME = plantdoc-container
PORT = 8000

# Build l'image Docker
docker_build:
	docker build -t $(IMAGE_NAME) .

# Test en local - run le container
docker_run_local:
	docker run -d -p $(PORT):$(PORT) --name $(CONTAINER_NAME) $(IMAGE_NAME)

# Stop et remove le container local
docker_stop:
	docker stop $(CONTAINER_NAME) && docker rm $(CONTAINER_NAME)

# Pour le cloud (exemple avec Google Cloud Run)
docker_deploy_cloud:
	docker tag $(IMAGE_NAME) gcr.io/YOUR_PROJECT_ID/$(IMAGE_NAME)
	docker push gcr.io/YOUR_PROJECT_ID/$(IMAGE_NAME)
	gcloud run deploy $(IMAGE_NAME) \
		--image gcr.io/YOUR_PROJECT_ID/$(IMAGE_NAME) \
		--platform managed \
		--region europe-west1 \
		--allow-unauthenticated

# Ou pour AWS ECR + ECS
docker_deploy_aws:
	aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com
	docker tag $(IMAGE_NAME) YOUR_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/$(IMAGE_NAME)
	docker push YOUR_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/$(IMAGE_NAME)

# Tout en un : build + run local
docker_local: docker_build docker_run_local
	@echo "API running on http://localhost:$(PORT)"
	@echo "Docs available at http://localhost:$(PORT)/docs"
