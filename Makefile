PROJECT := pv-yolov8

.PHONY: help setup data lint test train-local images docker-verify push \
	ecr pipeline pipeline-upsert deploy monitor cicd events destroy clean

help: ## Show targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

setup: ## Create .venv and install deps
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
	@echo "Activate with: source .venv/bin/activate"

data: ## Unpack person.zip + vehicle.zip into data/raw
	.venv/bin/python src/data/download.py

lint: ## Ruff check
	.venv/bin/ruff check src pipelines scripts tests

test: ## Unit tests (no AWS, no GPU)
	.venv/bin/pytest -q -m "not slow and not docker"

test-all: ## All tests including slow (torch/ultralytics)
	.venv/bin/pytest -q

train-local: data ## Preprocess + train + evaluate on this machine
	.venv/bin/python src/processing/preprocess.py
	.venv/bin/python src/training/train.py
	.venv/bin/python src/processing/evaluate.py

images: ## Build all Docker images locally (no push)
	./scripts/build_and_push.sh --skip-push

docker-verify: images ## Smoke-test containers against /opt/ml layout
	./scripts/verify_containers.sh

push: ## Build and push images to ECR; write build/images.json
	./scripts/build_and_push.sh
	.venv/bin/python scripts/resolve_digests.py

ecr: ## Deploy ECR repositories
	aws cloudformation deploy --template-file infra/ecr.yaml \
		--stack-name $(PROJECT)-ecr --capabilities CAPABILITY_NAMED_IAM

pipeline: ## Upsert + start SageMaker Pipeline and wait
	.venv/bin/python pipelines/training_pipeline.py --upsert --start --wait

pipeline-upsert: ## Upsert pipeline definition only
	.venv/bin/python pipelines/training_pipeline.py --upsert

deploy: ## Deploy latest Approved model package (needs SAGEMAKER_ROLE_ARN, ARTIFACT_BUCKET)
	.venv/bin/python pipelines/deploy.py \
		--role-arn "$(SAGEMAKER_ROLE_ARN)" \
		--bucket "$(ARTIFACT_BUCKET)" \
		--alarm-topic-arn "$(ALARM_TOPIC_ARN)"

monitor: ## Create drift alarm + CloudWatch dashboard
	.venv/bin/python pipelines/monitoring.py --topic-arn "$(ALARM_TOPIC_ARN)"

cicd: ## Deploy CodePipeline stack
	aws cloudformation deploy --template-file infra/codepipeline.yaml \
		--stack-name $(PROJECT)-cicd --capabilities CAPABILITY_NAMED_IAM

events: ## Deploy EventBridge monthly + approval + drift rules
	aws cloudformation deploy --template-file infra/eventbridge.yaml \
		--stack-name $(PROJECT)-events --capabilities CAPABILITY_NAMED_IAM

destroy: ## Tear down endpoint / schedules / stacks (keeps ECR)
	-aws cloudformation delete-stack --stack-name $(PROJECT)-endpoint
	-aws cloudformation delete-stack --stack-name $(PROJECT)-events
	-aws cloudformation delete-stack --stack-name $(PROJECT)-cicd
	@echo "ECR stack $(PROJECT)-ecr retained (inference digests must outlive endpoints)"

clean: ## Remove local artifacts and caches
	rm -rf data/raw data/processed artifacts build .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
