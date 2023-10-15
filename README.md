# Ask-picturize-it

[![licenses](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This uses OpenAI API Whisper(whisper-1), DALL-E, GPT(gpt-3.5-turbo), also Azure OpenAI and Google PaLM (https://developers.generativeai.google)

![image](https://github.com/amitpuri/Ask-picturize-it/assets/6460233/c25cf323-3102-4828-91fb-c5e4e3c329ed)

![image](https://github.com/amitpuri/Ask-picturize-it/assets/6460233/04dc970f-dea2-4b0f-9aec-67af4ff1073f)


## Azure Container Registry and Docker notes

export required environment variables

    export AZURE_REGISTRY_NAME = "Set Azure Container registry name here"
    export P_MONGODB_DATABASE = "Mongo database"
    export P_MONGODB_URI = "Mongo connection string"

build a docker image locally or use [GitHub Workflow action](.github/workflows)

    docker build --rm --pull \
      --file "Dockerfile" \
      --label "com.$AZURE_REGISTRY_NAME.ask-picturize-it" \
      --tag "ask-picturize-it:latest" \
      .

Run and test 

    docker run -e P_MONGODB_DATABASE -e P_MONGODB_URI -it --publish 80:80 --publish 27017:27017 ask-picturize-it:latest

Login to Azure Container registry

    az login --use-device-code
    az acr login --name $AZURE_REGISTRY_NAME.azurecr.io

 Tag and Push the docker image to the Azure Container registry   
 
    docker tag ask-picturize-it $AZURE_REGISTRY_NAME.azurecr.io/ask-picturize-it
    docker push $AZURE_REGISTRY_NAME.azurecr.io/ask-picturize-it


## JupyterLab or Azure ML Workspace Notes

- Use [https://jupyter.org](https://jupyter.org) (locally, or on the Cloud or in GitHub Codespaces) or Login to [https://ml.azure.com](https://ml.azure.com) and use Notebooks
- Clone repo and rename env.example to .env and set values to variables
- uncomment following lines
  
        #from dotenv import load_dotenv
        #load_dotenv()

- Install
  
        pip install python-dotenv

- Run to install dependencies

      pip install -r requirements.txt 
  
- Install pip install p2j
  
      pip install p2j
  
- Run to convert py to ipynb
  
      p2j app.py
