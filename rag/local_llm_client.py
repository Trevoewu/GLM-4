import requests
import json
import logging
from typing import Dict, Any, Optional

# 自定义JSON编码器处理StringPromptValue
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_string'):
            return obj.to_string()
        elif hasattr(obj, 'text'):
            return obj.text
        return super().default(obj)

logger = logging.getLogger(__name__)

class LocalGLM4Client:
    """本地GLM4模型客户端"""
    
    def __init__(self, base_url: str = None):
        if base_url is None:
            try:
                from config import LOCAL_GLM4_URL
                self.base_url = LOCAL_GLM4_URL
            except ImportError:
                self.base_url = "http://localhost:8001"
        else:
            self.base_url = base_url
        self.session = requests.Session()
    
    def generate_response(self, prompt: str, max_tokens: int = 2048, 
                         temperature: float = 0.1, stream: bool = False) -> str:
        """生成回复"""
        try:
            # 处理 StringPromptValue 对象
            if hasattr(prompt, 'to_string'):
                prompt_str = prompt.to_string()
            elif hasattr(prompt, 'text'):
                prompt_str = prompt.text
            else:
                prompt_str = str(prompt)
                
            # 构建请求数据
            data = {
                "model": "glm4-9b",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_str
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
            
            # 发送请求，增加超时时间
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=data,
                timeout=120,  # 增加到120秒
                stream=stream
            )
            
            if response.status_code == 200:
                if stream:
                    return self._handle_stream_response(response)
                else:
                    result = response.json()
                    if result.get("choices") and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                    else:
                        logger.error("响应中没有找到choices")
                        return "抱歉，无法生成回复。"
            else:
                logger.error(f"请求失败，状态码: {response.status_code}")
                return f"请求失败，状态码: {response.status_code}"
                
        except Exception as e:
            logger.error(f"生成回复时出错: {e}")
            return f"生成回复时出错: {str(e)}"
    
    def _handle_stream_response(self, response) -> str:
        """处理流式响应"""
        import json
        import sys
        
        full_response = ""
        try:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # 移除 'data: ' 前缀
                        if data == '[DONE]':
                            break
                        try:
                            json_data = json.loads(data)
                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                delta = json_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_response += content
                                    print(content, end='', flush=True)
                        except json.JSONDecodeError:
                            continue
                    elif line.strip():  # 处理非data行
                        try:
                            json_data = json.loads(line)
                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                delta = json_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_response += content
                                    print(content, end='', flush=True)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"处理流式响应时出错: {e}")
        
        print()  # 换行
        return full_response
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False

from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import LLMResult, Generation
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional, Iterator
import asyncio

class MockLocalLLM(BaseLanguageModel):
    """模拟本地LLM，兼容LangChain新版接口"""
    
    def __init__(self, client: Optional['LocalGLM4Client'] = None):
        super().__init__()
        self._client = client

    @property
    def _llm_type(self) -> str:
        return "mock_local_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # 处理 StringPromptValue 对象
        if hasattr(prompt, 'to_string'):
            prompt_str = prompt.to_string()
        elif hasattr(prompt, 'text'):
            prompt_str = prompt.text
        else:
            prompt_str = str(prompt)
            
        if self._client:
            try:
                # 检查是否启用流式输出
                stream = kwargs.get('stream', False)
                return self._client.generate_response(prompt_str, stream=stream)
            except Exception as e:
                return self._get_fallback_response(prompt_str)
        else:
            return self._get_fallback_response(prompt_str)

    def _get_fallback_response(self, prompt: str) -> str:
        if "上市公司" in prompt and "监管" in prompt:
            return "根据相关法律法规，上市公司需要遵守严格的监管要求，包括信息披露、公司治理、财务报告等方面的规定。"
        elif "信息披露" in prompt:
            return "信息披露是上市公司的重要义务，包括定期报告和临时报告，确保投资者能够及时、准确、完整地了解公司信息。"
        elif "独立董事" in prompt:
            return "独立董事是上市公司治理结构的重要组成部分，负责监督公司运作，保护中小股东利益，确保公司合规经营。"
        else:
            return "根据提供的文档内容，我可以为您提供相关信息。请具体说明您想了解的法律法规问题。"

    # 新版LangChain要求的抽象方法补全
    def predict(self, text: str, **kwargs: Any) -> str:
        return self._call(text)

    async def apredict(self, text: str, **kwargs: Any) -> str:
        return self._call(text)

    def predict_messages(self, messages: List[Any], **kwargs: Any) -> str:
        prompt = "\n".join([m.content for m in messages])
        return self._call(prompt)

    async def apredict_messages(self, messages: List[Any], **kwargs: Any) -> str:
        prompt = "\n".join([m.content for m in messages])
        return self._call(prompt)

    def generate_prompt(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
        generations = [[Generation(text=self._call(prompt))] for prompt in prompts]
        return LLMResult(generations=generations)

    async def agenerate_prompt(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
        generations = [[Generation(text=self._call(prompt))] for prompt in prompts]
        return LLMResult(generations=generations)

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        return self._call(str(input))

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        return self._call(str(input)) 