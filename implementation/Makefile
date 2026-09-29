.PHONY: help setup test extract-clip extract-vljepa index retrieve report evaluate compare demo docker-build docker-up

help:
	@echo "setup         - install dependencies (CPU torch)"
	@echo "test          - run unit tests"
	@echo "extract       - extract embeddings for a model/dataset: make extract MODEL=clip DATASET=ucf_crime"
	@echo "index         - build FAISS index:        make index MODEL=clip DATASET=ucf_crime"
	@echo "retrieve      - run a text query:         make retrieve MODEL=clip DATASET=ucf_crime QUERY='a person running'"
	@echo "report        - generate LLM report:      make report MODEL=clip DATASET=ucf_crime QUERY='a fight'"
	@echo "evaluate      - run evaluation:           make evaluate MODEL=clip DATASET=ucf_crime"
	@echo "compare       - compare both models:      make compare"
	@echo "demo          - launch streamlit demo"
	@echo "docker-build  - build the deployment image"
	@echo "docker-up     - start the demo container"

MODEL ?= clip
DATASET ?= ucf_crime
QUERY ?= a person running
CONFIG ?= config/config.yaml

setup:
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt

test:
	python -m pytest tests -v

extract:
	python scripts/extract_embeddings.py --model $(MODEL) --dataset $(DATASET) --config $(CONFIG)

index:
	python scripts/build_index.py --model $(MODEL) --dataset $(DATASET) --config $(CONFIG)

retrieve:
	python scripts/run_retrieval.py --model $(MODEL) --dataset $(DATASET) --query "$(QUERY)" --config $(CONFIG)

report:
	python scripts/generate_report.py --model $(MODEL) --dataset $(DATASET) --query "$(QUERY)" --config $(CONFIG)

evaluate:
	python scripts/run_evaluation.py --model $(MODEL) --dataset $(DATASET) --config $(CONFIG)

compare:
	python scripts/compare_models.py --config $(CONFIG)

demo:
	streamlit run demo/streamlit_app.py

docker-build:
	docker compose build

docker-up:
	docker compose up demo
