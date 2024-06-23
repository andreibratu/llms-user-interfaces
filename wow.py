from openai import OpenAI

from configuration import APP_CONFIG


OPENAI_CLIENT = OpenAI(api_key=APP_CONFIG.openai.api_key)

print(OPENAI_CLIENT.fine_tuning.jobs.retrieve('ftjob-uBDvxyDzz6rsj16Qi6YKHE1S').error)