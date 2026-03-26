"""
Tool management infrastructure for LLM function calling.

Provides:
- ToolRegistry: Discover and register tools
- ToolExecutor: Execute tools based on LLM tool calls
- tool decorator: Define tools with OpenAI-compatible schema
"""

import json
import logging
import inspect
from typing import Any, Callable, Dict, List, Optional
import importlib

logger = logging.getLogger(__name__)


class ToolDefinition:
    """Represents a tool definition in OpenAI format."""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry for discovering and managing tools."""
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._modules_loaded: set = set()
    
    def register(self, tool_def: ToolDefinition) -> None:
        """Register a tool definition."""
        if tool_def.name in self._tools:
            logger.warning(f"Tool '{tool_def.name}' already registered, overwriting")
        self._tools[tool_def.name] = tool_def
        logger.debug(f"Registered tool: {tool_def.name}")
    
    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all(self) -> Dict[str, ToolDefinition]:
        """Get all registered tools."""
        return self._tools.copy()
    
    def get_tools(self, tool_refs: List[str]) -> List[ToolDefinition]:
        """
        Get tools by references.
        
        Args:
            tool_refs: List of tool references like:
                - "math.calculate" (module.function)
                - "calculate" (just function name)
                - "math" (all tools from module)
        
        Returns:
            List of ToolDefinition objects
        """
        tools = []
        
        for ref in tool_refs:
            if "." in ref:
                # Module.function or module format
                parts = ref.split(".", 1)
                module_name = parts[0]
                function_name = parts[1] if len(parts) > 1 else None
                
                # Load module if not already loaded
                if module_name not in self._modules_loaded:
                    module_path = f"easy_scaffold.tools.{module_name}"
                    self._load_module(module_path)
                    self._modules_loaded.add(module_name)
                
                if function_name:
                    # Specific function
                    tool = self.get(function_name)
                    if tool:
                        tools.append(tool)
                    else:
                        logger.warning(f"Tool '{function_name}' not found in module '{module_name}'")
                else:
                    # All tools from module
                    module_tools = [
                        tool for name, tool in self._tools.items()
                        if tool.function.__module__ == f"easy_scaffold.tools.{module_name}"
                    ]
                    tools.extend(module_tools)
            else:
                # Just function name
                tool = self.get(ref)
                if tool:
                    tools.append(tool)
                else:
                    logger.warning(f"Tool '{ref}' not found")
        
        return tools
    
    def _load_module(self, module_path: str) -> None:
        """Dynamically load a tool module."""
        try:
            importlib.import_module(module_path)
            logger.debug(f"Successfully loaded tool module: {module_path}")
        except ImportError as e:
            logger.warning(f"Failed to load tool module '{module_path}': {e}")
    
    def to_openai_format(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert list of tools to OpenAI format."""
        return [tool.to_openai_format() for tool in tools]


class ToolExecutor:
    """Execute tools based on LLM tool calls."""
    
    def __init__(self, tools: List[ToolDefinition]):
        self._tools = {tool.name: tool for tool in tools}
    
    async def execute(self, tool_call: Any) -> Dict[str, Any]:
        """
        Execute a single tool call.
        
        Args:
            tool_call: Tool call object from LLM response (has .function.name, .function.arguments, .id)
        
        Returns:
            Tool response in OpenAI format
        """
        # Extract tool call info (handle both dict and OpenAI SDK formats)
        if hasattr(tool_call, 'function'):
            function_name = tool_call.function.name
            arguments_str = tool_call.function.arguments
            tool_call_id = tool_call.id
        elif isinstance(tool_call, dict):
            function_name = tool_call.get("function", {}).get("name")
            arguments_str = tool_call.get("function", {}).get("arguments", "{}")
            tool_call_id = tool_call.get("id")
        else:
            raise ValueError(f"Unknown tool_call format: {type(tool_call)}")
        
        if function_name not in self._tools:
            error_msg = f"Tool '{function_name}' not found"
            logger.error(error_msg)
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": json.dumps({"error": error_msg}),
            }
        
        tool_def = self._tools[function_name]
        
        try:
            # Parse arguments
            if isinstance(arguments_str, str):
                arguments = json.loads(arguments_str)
            else:
                arguments = arguments_str
            
            # Execute tool function
            if inspect.iscoroutinefunction(tool_def.function):
                result = await tool_def.function(**arguments)
            else:
                result = tool_def.function(**arguments)
            
            # Serialize result
            if isinstance(result, (dict, list)):
                content = json.dumps(result)
            elif isinstance(result, str):
                content = result
            else:
                content = json.dumps({"result": str(result)})
            
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": content,
            }
            
        except Exception as e:
            error_msg = f"Error executing tool '{function_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": json.dumps({"error": error_msg}),
            }
    
    async def execute_all(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """Execute multiple tool calls and return results."""
        results = []
        for tool_call in tool_calls:
            result = await self.execute(tool_call)
            results.append(result)
        return results


# Global registry instance
_registry = ToolRegistry()


def tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
) -> Callable:
    """
    Decorator to register a tool.
    
    Args:
        name: Tool name (must be unique)
        description: Tool description for LLM
        parameters: JSON Schema for parameters (OpenAI format)
    
    Example:
        @tool(
            name="calculate",
            description="Evaluate math expression",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        )
        async def calculate(expression: str) -> float:
            return eval(expression)
    """
    def decorator(func: Callable) -> Callable:
        tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=func,
        )
        _registry.register(tool_def)
        
        # Preserve original function - return as-is, just register it
        func._tool_definition = tool_def
        return func
    
    return decorator


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _registry


