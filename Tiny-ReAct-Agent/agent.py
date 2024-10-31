import json5

from llm import GLM4Chat
from tool import Tools


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
(this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""


class Agent:
    def __init__(self, path: str = '') -> None:
        self.path = path
        self.tool = Tools()
        self.system_prompt = self.build_system_input()
        self.model = GLM4Chat(path)

    def build_system_input(self):
        tool_descs, tool_names = [], []
        for tool in self.tool.toolConfig:
            tool_descs.append(TOOL_DESC.format(**tool))
            tool_names.append(tool['name_for_model'])
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)
        sys_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
        return sys_prompt
    
    def parse_latest_func_call(self, text):
        i = text.rfind('\nAction:')
        j = text.rfind('\nAction Input:')
        k = text.rfind('\nObservation:')
        if k < j: 
            text = text.rstrip() + '\nObservation:'
            k = text.rfind('\nObservation:')
        func_name = text[i + len('\nAction:') : j].strip()
        func_args = text[j + len('\nAction Input:') : k].strip()
        text = text[:k]
        return func_name, func_args, text
    
    def call_func(self, func_name, func_args):
        func_args = json5.loads(func_args)
        if func_name == 'google_search':
            return '\nObservation:' + self.tool.google_search(**func_args)

    def text_completion(self, text, history=[]):
        text = "\nQuestion:" + text
        response, his = self.model.chat(text, history, self.system_prompt)
        print("第一次回复：" + response)
        func_name, func_args, response = self.parse_latest_func_call(response)
        if func_name:
            response += self.call_func(func_name, func_args)
            print("调用工具之后：" + response)
        response, his = self.model.chat(response, history, self.system_prompt)
        print("最后：" + response)
        return response, his

# if __name__ == '__main__':
#     agent = Agent('YOUR MODEL PATH')
#     prompt = agent.build_system_input()
#     agent.text_completion('2024年美国的总统是谁？')