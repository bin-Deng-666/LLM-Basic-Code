import requests

class Tools:
    def __init__(self) -> None:
        self.toolConfig = self._tools()
    
    def _tools(self):
        tools = [
            {
                'name_for_human': '谷歌搜索',
                'name_for_model': 'google_search',
                'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
                'parameters': [
                    {
                        'name': 'search_query',
                        'description': '搜索关键词或短语',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            }
        ]
        return tools

    def google_search(self, search_query: str):
        url = "YOUR GOOGLE SEARCH URL"
        
        headers = {}
        
        payload = {}
        response = requests.post(url, headers=headers, json=payload)
        response = response.json()

        return response
    
# if __name__ == '__main__':
#     tool = Tools()
#     print(tool.google_search('周杰伦'))
    