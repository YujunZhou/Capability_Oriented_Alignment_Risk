# verl/tools/dice_tools.py
import random
from typing import Dict, Any
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

class DiceTool(BaseTool):
    """Dice tool, provides dice rolling functionality"""
    
    def __init__(self, config: dict = None):
        """Initialize the dice tool"""
        if config is None:
            config = {}
        super().__init__(config, self.get_openai_tool_schema())
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Get the OpenAI function tool schema for this tool"""
        return OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "roll_dice",
                "description": "Roll a six-sided dice and return the result",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        )
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Roll the dice and return the result"""
        result = random.randint(1, 6)
        return {"result": result}