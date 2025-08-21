FASTAPI_IMAGE := fastapi-app
FASTAPI_CONTAINER := fastapi-container
FASTAPI_DOCKERFILE := Dockerfile
PORT := 8000
# build:
# 	@echo "Building Docker api image: $(FASTAPI_IMAGE)"
# 	docker build -f $(FASTAPI_DOCKERFILE) -t $(FASTAPI_IMAGE) .

# run:
# 	@echo "Running Docker api container..."
# 	docker run -d --rm --name fastapi-container  -p 8000:8000 $(FASTAPI_IMAGE)

# buildrun:
# 	@echo "Building Docker api image: $(FASTAPI_IMAGE)"
# 	docker build -f $(FASTAPI_DOCKERFILE) -t $(FASTAPI_IMAGE) .
# 	@echo "Running Docker api container..."
# 	docker run -d --rm --name fastapi-container  -p 8000:8000 $(FASTAPI_IMAGE)

# clean:
# 	@echo "Stopping and removing FastAPI container..."
# 	docker stop fastapi-container || true
# 	docker rm fastapi-container || true
# 	@echo "Stopping and removing Streamlit container..."
# 	docker stop streamlit-container || true
# 	docker rm streamlit-container || true
# 	@echo "Removing Docker images..."
# 	docker rmi $(FASTAPI_IMAGE) || true
# 	docker rmi $(STREAMLIT_IMAGE) || true
# 	@echo "Removing network app-network"
# 	docker network rm app-network

# # Can include -v for volume. Went over this in lecture 2. Whichever director youyo'ure working on gets mounted to the docker container. So if you make a change to the log file it gets modified docker
# # app folder 
build:
	@echo "Building Docker image: $(FASTAPI_IMAGE)"
	docker build -t $(FASTAPI_IMAGE) .

# run: This command should run a container from your image, mapping the container's port to a port on your local machine so you can access the app in your browser.
run:
	@echo "Running Docker container on port $(PORT)"
	docker run -d --name $(FASTAPI_CONTAINER) -p $(PORT):$(PORT) $(FASTAPI_IMAGE)

# clean: This command should delete the image.
clean:
	@echo "Stopping and removing container first"
	docker stop $(FASTAPI_CONTAINER) || true
	docker rm $(FASTAPI_CONTAINER) || true

	@echo "Removing Docker image: $(FASTAPI_IMAGE)"
	docker rmi $(FASTAPI_IMAGE) || true