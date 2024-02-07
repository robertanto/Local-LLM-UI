from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

import json 
import warnings
from .hf_api import HuggingFaceInferenceAPI

class HuggingFaceInferenceLLM(LLM):
    model:Any

    def __init__(self, model_id, model_args={"max_new_tokens":100,"temperature":0.7}, api_token=None, *args, **kwargs):
        super(HuggingFaceInferenceLLM, self).__init__(*args, **kwargs)
        self.model = HuggingFaceInferenceAPI(
            model_id=model_id,
            model_args=model_args,
            api_token=api_token
        )
        
    @property
    def _llm_type(self) -> str:
        return "hugging_face_api"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        if run_manager is not None:
            warnings.warn("Warning...........run_manager not used")
        
        return self.model.query(prompt=prompt)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_id": self.model.model_id, "model_args":json.dumps(self.model.model_args)}