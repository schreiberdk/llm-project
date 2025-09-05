hello:
	@echo 'It works!'

##################
## Install reqs ##
##################
install_requirements:
	@pip install -r requirements.txt


##################
#### Run API #####
##################
run_api:
	uvicorn fast_api.api:app --reload --port 8001


##################
#### Testing #####
##################
.PHONY: tests

test:
		pytest --maxfail=1 --disable-warnings -q


##################
# DOCKER SECTION #
##################
IMAGE_NAME=257402714841.dkr.ecr.eu-north-1.amazonaws.com/llm-project
TAG=latest-cpu

# Build Docker image
docker-build:
	docker build -t $(IMAGE_NAME):$(TAG) .

# Run Docker container locally
docker-run:
	docker run --gpus all -it --rm -p 8001:8001 $(IMAGE_NAME):$(TAG)

# Push to Docker Hub
docker-push:
	docker tag $(IMAGE_NAME):$(TAG) schreiberdk/$(IMAGE_NAME):$(TAG)
	docker push schreiberdk/$(IMAGE_NAME):$(TAG)

# Stop all running containers (optional)
docker-stop:
	docker stop $(shell docker ps -q)
