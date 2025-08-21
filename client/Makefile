IMAGE_NAME := streamlit-toxic-comment-app
CONTAINER_NAME := streamlit-client-container
PORT := 8501

# build: This command should build the Docker image and give it a name (e.g., sentiment-app).
build:
	@echo "Building Docker image: $(IMAGE_NAME)"
	docker build -t $(IMAGE_NAME) .

# run: This command should run a container from your image, mapping the container's port to a port on your local machine so you can access the app in your browser.
run:
	@echo "Running Docker container on port $(PORT)"
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):$(PORT) --env-file .env $(IMAGE_NAME)

# clean: This command should delete the image.
clean:
	@echo "Stopping and removing container first"
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

	@echo "Removing Docker image: $(IMAGE_NAME)"
	docker rmi $(IMAGE_NAME) || true