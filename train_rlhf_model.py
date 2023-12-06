from textrl import TextRLEnv,TextRLActor
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModelWithLMHead
import logging
import sys
import pfrl
import torch
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForTokenClassification
)
from transformers.pipelines import pipeline

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
reward_model = pipeline(
    "text-classification",
    model='/share/portal/apk74/rlhf/results_home/checkpoint-9837', 
    tokenizer = tokenizer,
    return_all_scores=True
)
def get_reward_from_text(text):
    rating_scores = reward_model(text)[0]
    rating_scores = {d['label']: d['score'] for d in rating_scores}
    return rating_scores['LABEL_0'] * 10 ## Corresponds to 5 star ratings

class MyRLEnv(TextRLEnv):
    def get_reward(self, input_item, predicted_list, finish): # predicted will be the list of predicted token
    #   reward = 0
    #   if finish or len(predicted_list) >= self.env_max_length:
      predicted_text = tokenizer.convert_tokens_to_string(predicted_list[0])
    #     # sentiment classifier
      reward = get_reward_from_text(predicted_text)
      return reward

tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-rw-1b')
model = AutoModelForCausalLM.from_pretrained('tiiuae/falcon-rw-1b').to("cuda")
model.eval()
observation_list = [{"input":
            'Home and Kitchen product review: '},
            {"input":
            'Review: '}
        ]
env = MyRLEnv(model, tokenizer, observation_input=observation_list,compare_sample=1, max_length= 100)
actor = TextRLActor(env,model,tokenizer,top_p = 0.3,act_deterministically=False)
agent = actor.agent_ppo(update_interval=100, minibatch_size=3, epochs=10)

print(actor.predict(observation_list[0]))


pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=4000,
    eval_n_steps=None,
    eval_n_episodes=3,       
    train_max_episode_len=200,  
    eval_interval=10,
    outdir='1_star_reviews_v2', 
)