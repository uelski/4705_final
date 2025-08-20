# Toxic Comment Moderation App
This project includes an api directory, client directory, models directory, and monitoring directory for our Toxic Comment Moderation app as part of the comp_4705 final project.

FastAPI use

Build and run the container:
$make buildrun
## API use with Postman when using a local Docker container:
Post requests to http://127.0.0.1:8000/predict
1. With no true labels included:
{
    "text": "this is a toxic comment"
}
### Response (and sent to DynamoDB log)
{
    "timestamp": "2025-08-20T19:00:56.219755",
    "request_text": "this is a toxic comment",
    "response": {
        "toxic": 1,
        "severe_toxic": 0,
        "obscene": 0,
        "threat": 1,
        "insult": 1,
        "identity_hate": 1
    },
    "true_labels": null
}

2. With a single True label included:
{
    "text": "this is a toxic comment",
    "true_labels": {
        "toxic": 1
    }
}
### Response (and sent to DynamoDB log)
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
        "toxic": 1
    }
}

3. With multiple unordered True labels included:
{
    "text": "this is a toxic comment",
    "true_labels": {
        "identity_hate": 0,
        "toxic": 1
    }
}
### Response (and sent to DynamoDB log)
{
    "timestamp": "2025-08-20T19:49:51.032089",
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
        "identity_hate": 0,
        "toxic": 1
    }
}

DynamoDB Table Creation & Connection
Table name: table_01
Partition key: "timestamp" type = String

EC2 Creation for FastAPI
Download and install putty  
Create the EC2 instance:
    1. Launch and Configure EC2 Instance:
        ○ Launch a new t2.micro EC2 instance with Ubuntu.
        Create a new key *during* the EC2 instance creation and save it
        ○ Configure its Security Group to allow incoming traffic on:
            § Port 22 (SSH) from your IP address for access.
            § Port 8000 (FastAPI) from anywhere (Custom TCP)
            Set IAM role to LabInstanceProfile
        ○ Connect to your EC2 instance using SSH.
    2. Set up the Server Environment:
        ○ On the EC2 instance, install Docker and Git.
    3. Deploy the Application:
        ○ Clone your GitHub repository onto the EC2 instance.
        ○ Create a shared Docker volume for the logs.
        ○ Build the Docker images for both the api and monitoring services.
Run both containers in detached mode (-d), ensuring they are connected to the shared volume.

For EC2 SSH access
Download putty
Create a new key *during* the EC2 instance creation
Download it to a directory
Right click them pem file and open with puttygen.exe
Click save private key in the same directory as the pem
Open putty.exe
Copy the public ipv4 address into the Host Name field in putty
In putty, Under Connection>SSH>Auth>Credentials in the "Private key file for authentication" click browse and open the ppk
In putty, under Window>Appearance, increase the font size.
In putty, Under Session, save this session
In putty, click open

In the putty cli, the username is "ubuntu". 
Run each of the following commands:

$ sudo apt-get update -y
$ sudo apt-get install docker.io -y

$ sudo systemctl start docker
$ sudo systemctl enable docker
$ sudo usermod -aG docker ubuntu
$ git --version (to confirm git is installed)

Cloning github (instructions for specific branch and directory)
$ git clone --no-checkout --depth 1 --branch feature/api-v02 https://github.com/uelski/4705_final.git
$ cd 4705_final
$ git sparse-checkout init --cone
$ git sparse-checkout set api
$ git checkout feature/api-v02
$ cd api

create secrets.json:
$ echo '{ "wandb_api_key": "1234" }' > secrets.json

Running the app:
$ sudo apt install make
$ sudo apt install make-doc
$ sudo make buildrun


http://18.213.2.212:8000/predict