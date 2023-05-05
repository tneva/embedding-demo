Based on the code taken from the OpenAI embedding demo, I've just broken things up and refactored some code to avoid rate limiting when getting the embed vectors for each line in the `scraped.csv` file.

I've chosen to leave the processed files in as generating the embedded data can be quite time consuming, this makes it easier to see how things are working as you can just run `chat.py` to see how things work.

You need to set up an OpenAI secret key for your account: https://platform.openai.com/docs/api-reference/introduction to start with. If you want to start from scratch to see how the whole process works then you can delete the `processed` folder and then just run:

```
scrape.py to scrape the site
create-embeddings.py to parse the text files, flatten them all in to a csv file and then generate the embedding data
chat.py to start asking questions
```

or you can just keep the processed folder and do `chat.py`

The processed data was generated on the day of the first commit for this repo

You can set you Python env with:

```
python -m venv env

source env/bin/activate

pip install -r requirements.txt
```