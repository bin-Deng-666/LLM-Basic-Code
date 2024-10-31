from typing import List

from transformers import AutoModel, AutoTokenizer

class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str):
        pass

    def load_model(self):
        pass

    
class GLM4Chat(BaseModel):
    def __init__(self, path: str= '')-> None:
        super().__init__(path)
        self.load_model()
        self.gen_kwargs = {"max_new_tokens": 8192, "num_beams": 1, "do_sample": True, "top_p": 0.8, "temperature": 0.8, "eos_token_id": self.model.config.eos_token_id}
        
    def load_model(self):
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True, encode_special_tokens=True)
        self.model = AutoModel.from_pretrained(self.path, trust_remote_code=True, device_map="auto").eval()
        print('================ Model loaded ================')
        
    def chat(self, prompt: str, history: List[dict], system_prompt: str=''):
        history += [{"role": "user", "content": prompt}]
        message = [{"role": "system", "content": system_prompt}] + history
        
        input_message = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(input_message, return_tensors="pt", padding="max_length", truncation=True, max_length=len(input_message)).to(self.model.device)
        outputs = self.model.generate(**inputs, **self.gen_kwargs)[:, inputs['input_ids'].shape[1]:]
        outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        history += [{"role": "assistant", "content": outputs}]
        return outputs, history
        
# if __name__ == '__main__':
#     model = GLM4Chat('YOUR MODEL PATH')
#     print(model.chat('Hello, can you introduce yourself?', []))