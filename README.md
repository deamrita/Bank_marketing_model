# Bank_marketing_model
Building a classification model to predict the outcome of a bank marketing phone call.

Tech stack: python (pandas, numpy, sklearn, streamlit, seaborn, matplotlib)

For deployment: https://streamlit.io/cloud

App link: https://bankmarketingmodel-hf2uviq2z7w.streamlit.app/

Steps:
1. Built the model in Jupyter Notebook in the file named "Bank marketing End to End Project.py".
2. Create a new python file app.py to create the webapp.
3. put all the required package names (pandas, numpy, streamlit, scikit-learn) in the file requirements.txt.
4. Put all the folders in the same directory.
5. Created a new environment with Streamlit.
   a. Go to the folder where your project will live. cd "project location"
   b. Create a new Pipenv environment in that folder: python -m venv venv_bnk
   When you run the command above, a directory called venv will appear in myprojects/. This directory is where your Python environment and dependencies are installed.
6. Activate your environment: venv_bnk\Scripts\activate
7. run the command: pip install -r requirements.txt (onetime run)
8. run the command: streamlit run app.py
9. Streamlit will start a local development server and launch your app in a web browser.
10. Test your application: You can now interact with your Streamlit app in the web browser. Enter the feature values, click the "Predict" button, and view the predicted output.
11. Deploy to Streamlit Sharing (streamlit.io): Streamlit Sharing (formerly known as Streamlit for Teams) allows you to deploy and share your Streamlit app online. To deploy your app to Streamlit Sharing, follow these steps:
 a. Create an account on Streamlit Sharing (https://streamlit.io/sharing).
 b. Deploy through your Github profile.
   
