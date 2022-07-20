# Faces_Emotions

## Introduction
The data consists of 20x20 pixels grayscale images of faces. The faces have been automatically registered.
The task is to categorize each face based on the emotion shown in the facial expression into one of six categories (0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Surprise, 5=Neutral). The training set consists of 33,000 examples

## Model Card

The model was deployed to the web using the FastAPI package and API tests were created. The API tests will be embedded in a CI/CD framework using GitHub Actions. After we built our API locally and tested it, we deployed it to Heroku and tested it again live. Weights and Biases were used to manage and track all artifacts.

<img align="center" src="https://github.com/Morsinaldo/credit_risk_mlops/blob/main/images/Model_card.png" />

So, in general, the notebooks used were divided into 7 parts:

  1. The search for data
  2. Exploratory analysis
  3. Pre-Processing
  4. Tests
  5. Splitting the data between training and testing.
  6. Training
  7. Test


## Anaconda Environment

Create a conda environment with ``environment.yml``:

```bash
conda env create --file environment.yml
```

To remove an environment in your terminal window run:

```bash
conda remove --name myenv --all
```

To list all available environments run:

```bash
conda env list
```

To activate the environment, use

```bash
conda activate myenv
```

## Fast API

The API is implemented in the ``source/api/main.py`` whereas tests are on ``source/api/test_main.py``.

For the sake of understanding and during the development, the API was constanly tested using:

```bash
uvicorn source.api.main:app --reload
```

and using these addresses:

```bash
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
```

The screenshot below show a view of the API docs.

![image](https://user-images.githubusercontent.com/50224653/179810307-40c1738b-0244-4bf8-a8e8-666a8ad6f592.png)


For test the API, please run:

```bash
pytest source/api -vv -s
```

## Heroku

1. Sign up for free and experience [Heroku](https://signup.heroku.com/login).
2. Now, it's time to create a new app. It is very important to connect the APP to our Github repository and enable the automatic deploys.
3. Install the Heroku CLI following the [instructions](https://devcenter.heroku.com/articles/heroku-cli).
4. Sign in to heroku using terminal
```bash
heroku login
```
5. In the root folder of the project check the heroku projects already created.
```bash
heroku apps
```
6. Check buildpack is correct: 
```bash
heroku buildpacks --app faces-emotions
```
7. Update the buildpack if necessary:
```bash
heroku buildpacks:set heroku/python --app faces-emotions
```
8. When you're running a script in an automated environment, you can [control Wandb with environment variables](https://docs.wandb.ai/guides/track/advanced/environment-variables) set before the script runs or within the script. Set up access to Wandb on Heroku, if using the CLI: 
```bash
heroku config:set WANDB_API_KEY=xxx --app faces-emotions
```
9. The instructions for launching an app are contained in a ```Procfile``` file that resides in the highest level of your project directory. Create the ```Procfile``` file with:
```bash
web: uvicorn source.api.main:app --host=0.0.0.0 --port=${PORT:-5000}
```
10. Configure the remote repository for Heroku:
```bash
heroku git:remote --app faces-emotions
```
11. Push all files to remote repository in Heroku. The command below will install all packages indicated in ``requirements.txt`` to Heroku VM. 
```bash
git push heroku main
```
12. Check the remote files run:
```bash
heroku run bash --app faces-emotions
```
13. If all previous steps were done with successful you will see the message below after open: ```https://faces-emotions.herokuapp.com/```.
14. For debug purposes whenever you can fetch your app’s most recent logs, use the [heroku logs command](https://devcenter.heroku.com/articles/logging#view-logs):
```bash
heroku logs
```

## About me
I'm a student of the Postgraduate Program in Electrical and Computer Engineering (PPgEEC) at the Federal University of Rio Grande do Norte (UFRN). As the second project of the [EEC1509](https://github.com/ivanovitchm/ppgeecmachinelearning) — Machine Learning course taught by [Ivanovitch Silva](https://github.com/ivanovitchm), the project consists in take a classic neural network model and adapted it to a pipeline, which contains good standardization practices in order to put the created model into production.

## References

Main reference - [Ivanovitch's git repo](https://github.com/ivanovitchm/colab2mlops)

[Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

