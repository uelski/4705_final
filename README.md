# Toxic Comment Moderation App
This project includes an api directory, client directory, models directory, and monitoring directory for our Toxic Comment Moderation app as part of the comp_4705 final project.

## Dataset
Our dataset was pulled from the Kaggle ["Toxic Comment Classification Challenge"](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview). It contains text comments as the input and six separate toxicity labels as the output. A 1 indicates the toxicity label is associated with the comment per row, a 0 indicates it is not. For example:

```
comment_text  toxic  severe_toxic  obscene  threat  insult  identity_hate
text here      1             0        0       0       0              0
```
We decided to use three separate ML models to train and use to predict the labels given the input text:
- MultinomialNB
- LogisticRegression
- LinearSVC

## Architecture
This repository contains four separate directories, each a part of the overall project:
- "/api": The FastAPI backend that pulls in the ML model and has an endpoint to take in text and true labels, and output a prediction. This includes code to send logs for each time the prediction endpoint is hit to an AWS DynamoDB table.
- "/client": The frontend Streamlit application with a UI that allows users to input text and select any of 6 toxicity levels then submit to the api to analyze and view the predictions.
- "/models": The code to pull in the dataset from an AWS S3 bucket, train three separate ML models and compare metrics with the test set. This includes code to log the metrics for each model to W&B and build the models as artifacts to send to W&B, tagging as 'candidate' models to potentially be promoted to staging and production use.
- "/monitoring": The frontend Streamlit dashboard application that is connected to the same DynamoDB table as the api and pulls in the logs created from the /predict endpoint to use to visualize and display specific metrics including the distribution of prediction labels, and accuracy and precision.

We included a GitHub Actions file to run a linter and test files on all four subdirectories when a pull request has been submitted into the 'main' branch.

## Prerequisites
To get this app up and running Docker and python must be installed on your machine. Postman or 'curl' commands can be used to test the endpoints.

## Local Development
- 'git clone' this repo in project directory of choice.
- 'cd' into this cloned repo.
- API:
    - cd /api
    - create a secrets.json file with a working wandb.ai api key, populated like this:
        ```
        { "wandb_api_key": "<your_api_key>" }
        ```
    - Run 'make build' to build the Docker image.
    - Run 'make run' to run the Docker container
    - The endpoints should now be accessible at http://127.0.0.1:8000 on your machine.
    - Use Postman or curl commands to access and test the API endpoints.
    - Endpoint documentation can be accessed at http://127.0.0.1:8000/docs.
    - Run 'make clean' to remove the Docker image.
- CLIENT:
    - cd /client
    - create a .env file if you wish to test a deployed version of the api and populate it with the endpoint url like so:
        ```
        API_URL=http://<FASTAPI-EC2-PUBLIC-IP>:8000/predict
        ```
    - Run 'make build' to build the Docker image.
    - Run 'make run' to run the Docker container
    - Navigate to http://localhost:8501/ to view the frontend streamlit app
    - Run 'make clean' to remove the Docker image
    - Note: only one Streamlit app can be running and viewable at this port on a single machine.
- MODELS:
    - cd /models
    - run 'python model_training.py' to run the three separate model pipelines, log the metrics and the model artifacts to W&B
- MONITORING:
    - cd /monitoring
    - Run 'make build' to build the Docker image.
    - Run 'make run' to run the Docker container
    - Navigate to http://localhost:8501/ to view the frontend streamlit app
    - Use the dummy_logs.json file locally to test the visualization and metrics display.
    - Run 'make clean' to remove the Docker image
    - Note: only one Streamlit app can be running and viewable at this port on a single machine.

## How To Test
With Curl:<br>
- `curl http://127.0.0.1:8000/health`
- `curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"text": "Toxic Comment Here!", "true_labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}}'`

With Postman:<br>
- GET request
- url: "http://127.0.0.1:8000/health"
- POST request
- url: "http://127.0.0.1:8000/predict"
- body:
    - type: raw (JSON)
    - {"text": "Toxic Comment Here!", "true_labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}}

## Manual Deployment Guide
These are the directions to create AWS EC2 instances, connect via ssh, and build and run the API, CLIENT, and MONITORING applications from this repository. The high-level overview is you need to create 3 separate EC2 instances, ssh into each of them from your local machine and clone and pull this repository. Then you can change into the /api, /client, and /monitoring directories in the separate EC2 instances to build and run each application using the commands below.

Create an EC2 Instance:
- In the AWS Management Console search and navigate to the EC2 console.
- Select launch instance
- Give your instance a name - ie Monitoring App
- Select the Amazon Linux 2023 AMI or Ubuntu under the Application and OS Images section
- Under instance type select t2.micro
- Under Key pair (login) select a key pair that you already have a downloaded .pem file or putty setup for.
- This step is important because you will need it to ssh into the ec2 instance.
- Under Network Settings click the Edit button on the top right to set security group details. 
- For all three applications:
    - Set type: SSH, Protocol/Port: TCP 22, Source: My IP (auto-fills your public IPv4)
- For API:
    - Set type: Custom TCP, Port range: 8000, Source type: Anywhere, Source (IPv4): 0.0.0.0/0
- For CLIENT and MONITORING:
    - Set type: Custom TCP, Port range: 8501, Source type: Anywhere, Source (IPv4): 0.0.0.0/0
- Finally, select 'Launch Instance'

Now you can connect to your instances, download the necessary packages, git clone, and run the applications in your EC2 instances.
- navigate to the instance details by going to EC2 dashboard and clicking on the newly created instance
- view the public ipv4 address by clicking on the instance and viewing it's details. You will need this address later.

```
Note: there are multiple ways to ssh into and setup your EC2 instance. This guide will cover using a .pem file directly and Amazon Linux, and using putty with Ubuntu.
```

For using a .pem file directly with 'yum' commands and EC2 AMI Amazon Linux:
- go to a new terminal window on your machine and navigate to the directory you saved your .pem file
- run 'chmod 400 your-key.pem' to secure your key pair.
- run 'ssh -i your-key.pem ec2-user@\<EC2-Public-IP>' - replacing \<EC2-Public-IP> with your instance's public IP.
- this information to connect to your specific address is available in the AWS EC2 console at the EC2 instance under the 'Connect' option and 'SSH client'
- You should now be connected to your EC2 instance via the terminal window.
- Install Docker (https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-docker.html):
    - 'sudo yum update -y'
    - 'sudo yum install -y docker'
    - 'sudo service docker start'
    - 'sudo usermod -a -G docker ec2-user'
    - "Pick up the new docker group permissions by logging out and logging back in again. To do this, close your current SSH terminal window and reconnect to your instance in a new one. Your new SSH session should have the appropriate docker group permissions."
    - check permissions and docker working:
        - 'docker ps'
- install git:
    - 'sudo yum install git -y'
    - 'git --version'
- install make to run the Makefile:
    - 'sudo yum groupinstall "Development Tools"'

For using putty and 'apt-get' commands and EC2 AMI Ubuntu:
- Download putty
- Create a new key *during* the EC2 instance creation
- Download it to a directory
- Right click them pem file and open with puttygen.exe
- Click save private key in the same directory as the pem
- Open putty.exe
- Copy the public ipv4 address into the Host Name field in putty
- In putty, Under Connection>SSH>Auth>Credentials in the "Private key file for authentication" click browse and open the ppk
- In putty, under Window>Appearance, increase the font size.
- In putty, Under Session, save this session
- In putty, click open
- In the putty cli, the username is "ubuntu". 

Run each of the following commands:
- sudo apt-get update -y
- sudo apt-get install docker.io -y
- sudo systemctl start docker
- sudo systemctl enable docker
- sudo usermod -aG docker ubuntu
- git --version (to confirm git is installed)

DynamoDB Table Creation & Connection
- Navigate to DynamoDB dashboard in AWS
- Create new table
- Table name: table_01
- Partition key: "timestamp" type = String

Now you should have all of the necessary tools installed on your EC2 instances to run the applications.
- IMPORTANT: You will need to create files for the API and CLIENT applications within your EC2 instances before running them:
    - API:
    ```
    touch secrets.json
    echo '{ "wandb_api_key": "<your_api_key>" }' > secrets.json
    ```
    - CLIENT:
    ```
    touch .env
    echo 'API_URL=http://<FASTAPI-EC2-PUBLIC-IP>:8000/predict' > .env
    ```
- follow the 'Local Development' section of this readme:
    - git clone the repo on the EC2 instance
    - cd into the repo
    - cd into the specific target directory: /api, /client, or /monitoring
    - 'make build'
    - 'make run'
- to access the applications from your machine:
    - in the following replace \<EC2-PUBLIC-IP> with the public ipv4 address from above for each instance
    - navigate to http://\<EC2-PUBLIC-IP>:8501 for the CLIENT and MONITORING apps
    - navigate to http://\<EC2-PUBLIC-IP>:8000/docs to view the docs for the FastAPI app
    - in Postman interact with the endpoints located at:
        - GET http://\<EC2-PUBLIC-IP>:8000/health
        - POST http://\<EC2-PUBLIC-IP>:8000/predict
- to remove the Docker image from the EC2 instance run 'make clean'
- to exit the ssh connection run 'exit' in the terminal shell you are currently using to connect

That is it! You should now be able to connect to your EC2 instances via ssh and run the applications within this repository. 

## Example Request and Response

In Postman or through the frontend CLIENT interface an example request sent as a POST to the /predict endpoint is:
```
{"text": "Toxic Comment Here!", "true_labels": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}}
```

The API will then handle the request, parse the body and use the imported ML model to generate a prediction given the input "text" field. The API will then create a new object to send as a log to the DynamoDB table for tracking, as well as the response to the frontend to display both the predicted labels and the true labels the user sent along with the text:

```
{
    "timestamp": "2025-08-20T19:48:05.617747",
    "request_text": "this is a toxic comment",
    "response": {
        "toxic": 1,
        "severe_toxic": 0,
        "obscene": 0,
        "threat": 1,
        "insult": 1,
        "identity_hate": 1
    },
    "true_labels": {
        "toxic": 1, 
        "severe_toxic": 0, 
        "obscene": 0, 
        "threat": 0, 
        "insult": 0, 
        "identity_hate": 0
    }
}
```