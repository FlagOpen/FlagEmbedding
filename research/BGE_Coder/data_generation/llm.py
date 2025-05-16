import os
import time
import openai
import random
import tiktoken
import threading
from openai import OpenAI, AzureOpenAI
from typing import Tuple


class LLM:
    def __init__(
        self,
        model: str="Qwen2-5-Coder-32B-Instruct",
        model_type: str = "open-source",
        port: int = 8000,
    ):
        if model_type == "open-source":
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=f"http://localhost:{port}/v1/"
            )
        elif model_type == "azure":
            self.client = AzureOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION", "2024-02-01"),
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                azure_deployment=os.getenv("OPENAI_DEPLOYMENT_NAME", 'gpt-35-turbo')
            )
        elif model_type == "openai":
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", None)
            )
        else:
            raise ValueError("model_type must be one of ['open-source', 'azure', 'openai']")
        
        self.model = model
        self.tokenizer = tiktoken.get_encoding("o200k_base")
    
    def split_text(self, text: str, anchor_points: Tuple[float, float] = (0.4, 0.7)):
        token_ids = self.tokenizer.encode(text)
        anchor_point = random.uniform(anchor_points[0], anchor_points[1])
        split_index = int(len(token_ids) * anchor_point)
        return self.tokenizer.decode(token_ids[:split_index]), self.tokenizer.decode(token_ids[split_index:])
    
    def chat(
        self,
        prompt: str,
        max_tokens: int = 8192,
        logit_bais: dict = None,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.6,
        repetition_penalty: float = 1.0,
        remove_thinking: bool = True,
        timeout: int = 90,
    ):
        endure_time = 0
        endure_time_limit = timeout * 2
        
        def create_completion(results):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    logit_bias=logit_bais if logit_bais is not None else {},
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body={'repetition_penalty': repetition_penalty},
                    timeout=timeout,
                )
                results["content"] = [x.message.content for x in completion.choices[:n]]
            except openai.BadRequestError as e:
                # The response was filtered due to the prompt triggering Azure OpenAI's content management policy.
                results["content"] = [None for _ in range(n)]
            except openai.APIConnectionError as e:
                results["error"] = f'APIConnectionError({e})'
            except openai.RateLimitError as e:
                results["error"] = f'RateLimitError({e})'
            except Exception as e:
                results["error"] = f"Error: {e}"
        
        while True:
            results = {"content": None, "error": None}
            completion_thread = threading.Thread(target=create_completion, args=(results,))
            completion_thread.start()
            
            start_time = time.time()
            while completion_thread.is_alive():
                elapsed_time = time.time() - start_time
                if elapsed_time > endure_time_limit:
                    print("Completion timeout exceeded. Aborting...")
                    return [None for _ in range(n)]
                time.sleep(1)
            
            # If an error occurred during result processing
            if results["error"]:
                if endure_time >= endure_time_limit:
                    print(f'{results["error"]} - Skip this prompt.')
                    return [None for _ in range(n)]
                print(f"{results['error']} - Waiting for 5 seconds...")
                endure_time += 5
                time.sleep(5)
                continue
            
            content_list = results["content"]
            if remove_thinking:
                content_list = [x.split('</think>')[-1].strip('\n').strip() if x is not None else None for x in content_list]
            return content_list


if __name__ == "__main__":
    llm = LLM(
        model="gpt-4o-mini-2024-07-18",
        model_type="openai"
    )

    prompt = "hello, who are you?"
    response = llm.chat(prompt)[0]
    print(response)


if __name__ == "__main__":
    llm = LLM(
        model="gpt-4o-mini-2024-07-18",
        model_type="openai"
    )

    prompt = "hello, who are you?"
    response = llm.chat(prompt)[0]
    print(response)
