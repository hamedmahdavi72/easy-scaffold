"""
Sandbox implementations for safe code execution in tool calling.

Supports multiple sandbox backends:
- Modal: Cloud-based serverless sandbox
- Docker: Container-based local sandbox
- Restricted: RestrictedPython-based lightweight sandbox
- Subprocess: Basic process isolation (fallback)
"""

import asyncio
import json
import logging
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

# resource module is Unix-only (not available on Windows)
HAS_RESOURCE = sys.platform != "win32"
if HAS_RESOURCE:
    import resource

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    """Result from sandbox execution."""
    output: str
    error: Optional[str] = None
    success: bool = True
    execution_time: Optional[float] = None


class Sandbox(ABC):
    """Base interface for sandbox implementations."""
    
    @abstractmethod
    async def execute(
        self,
        code: str,
        timeout: int = 5,
        language: str = "python",
        **kwargs
    ) -> SandboxResult:
        """Execute code in sandboxed environment."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup sandbox resources (optional)."""
        pass


class ModalSandbox(Sandbox):
    """Modal cloud-based sandbox implementation."""
    
    def __init__(self, function_name: Optional[str] = None):
        """
        Initialize Modal sandbox.
        
        Args:
            function_name: Name of Modal function to call (if using pre-deployed function)
        """
        self.function_name = function_name
        self._modal_app = None
        self._function = None
        self._initialize_modal()
    
    def _initialize_modal(self):
        """Initialize Modal app and function."""
        try:
            import modal
            # Create or get Modal app
            self._modal_app = modal.App("olympiad-tools")
            
            # Define sandbox function
            @self._modal_app.function(
                image=modal.Image.debian_slim().pip_install(["sympy", "numpy"]),
                timeout=10,
                memory=512,
            )
            def execute_code_modal(code: str, timeout: int = 5) -> Dict[str, Any]:
                """Execute code in Modal's isolated environment."""
                import sys
                import io
                from contextlib import redirect_stdout, redirect_stderr
                
                # Restricted globals
                restricted_globals = {
                    "__builtins__": {
                        "abs": abs,
                        "round": round,
                        "min": min,
                        "max": max,
                        "sum": sum,
                        "pow": pow,
                        "range": range,
                        "len": len,
                        "str": str,
                        "int": int,
                        "float": float,
                        "bool": bool,
                    },
                }
                
                # Try to import safe modules
                try:
                    import math
                    restricted_globals["math"] = math
                except ImportError:
                    pass
                
                try:
                    import sympy
                    restricted_globals["sympy"] = sympy
                except ImportError:
                    pass
                
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                
                try:
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        result = eval(code, restricted_globals, {})
                    
                    return {
                        "success": True,
                        "output": str(result),
                        "stdout": stdout_capture.getvalue(),
                        "stderr": stderr_capture.getvalue(),
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "stdout": stdout_capture.getvalue(),
                        "stderr": stderr_capture.getvalue(),
                    }
            
            self._function = execute_code_modal
            logger.info("Modal sandbox initialized")
        except ImportError:
            logger.warning("Modal not installed. Install with: pip install modal")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Modal sandbox: {e}")
            raise
    
    async def execute(
        self,
        code: str,
        timeout: int = 5,
        language: str = "python",
        **kwargs
    ) -> SandboxResult:
        """Execute code using Modal."""
        if not self._function:
            raise RuntimeError("Modal sandbox not initialized")
        
        try:
            # Call Modal function
            result = await self._function.remote.aio(code, timeout=timeout)
            
            if result.get("success"):
                output = result.get("output", "")
                if result.get("stdout"):
                    output = f"{result['stdout']}\n{output}" if output else result["stdout"]
                return SandboxResult(
                    output=output,
                    success=True
                )
            else:
                return SandboxResult(
                    output="",
                    error=result.get("error", "Unknown error"),
                    success=False
                )
        except Exception as e:
            logger.error(f"Modal sandbox execution failed: {e}")
            return SandboxResult(
                output="",
                error=str(e),
                success=False
            )


class DockerSandbox(Sandbox):
    """Docker container-based sandbox implementation."""
    
    def __init__(self, image: str = "python:3.11-slim", memory_limit: str = "100m"):
        """
        Initialize Docker sandbox.
        
        Args:
            image: Docker image to use
            memory_limit: Memory limit (e.g., "100m", "512m")
        """
        self.image = image
        self.memory_limit = memory_limit
        self._client = None
        self._initialize_docker()
    
    def _initialize_docker(self):
        """Initialize Docker client."""
        try:
            import docker
            self._client = docker.from_env()
            # Test connection
            self._client.ping()
            logger.info(f"Docker sandbox initialized with image: {self.image}")
        except ImportError:
            logger.warning("Docker not installed. Install with: pip install docker")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Docker sandbox: {e}")
            raise
    
    async def execute(
        self,
        code: str,
        timeout: int = 5,
        language: str = "python",
        **kwargs
    ) -> SandboxResult:
        """Execute code in Docker container."""
        if not self._client:
            raise RuntimeError("Docker sandbox not initialized")
        
        import time
        start_time = time.time()
        
        try:
            # Escape code for shell - use base64 encoding for safety
            import base64
            code_bytes = code.encode('utf-8')
            code_b64 = base64.b64encode(code_bytes).decode('utf-8')
            command = f"python -c \"import base64; exec(base64.b64decode('{code_b64}').decode('utf-8'))\""
            
            # Run container in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def run_container():
                container = self._client.containers.run(
                    self.image,
                    command=command,
                    detach=True,
                    mem_limit=self.memory_limit,
                    cpu_period=100000,
                    cpu_quota=50000,  # 50% CPU
                    network_disabled=True,
                    remove=True,  # Auto-remove after execution
                )
                return container
            
            container = await loop.run_in_executor(None, run_container)
            
            # Wait for completion with timeout
            try:
                def wait_container():
                    return container.wait(timeout=timeout)
                
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, wait_container),
                    timeout=timeout + 1  # Slight buffer
                )
                exit_code = result.get("StatusCode", 1)
                
                # Get logs
                def get_logs():
                    return container.logs().decode('utf-8')
                
                logs = await loop.run_in_executor(None, get_logs)
                
                execution_time = time.time() - start_time
                
                if exit_code == 0:
                    return SandboxResult(
                        output=logs.strip(),
                        success=True,
                        execution_time=execution_time
                    )
                else:
                    return SandboxResult(
                        output="",
                        error=f"Container exited with code {exit_code}: {logs}",
                        success=False,
                        execution_time=execution_time
                    )
            except asyncio.TimeoutError:
                # Timeout - kill container
                def kill_container():
                    try:
                        container.kill()
                    except Exception:
                        pass
                
                await loop.run_in_executor(None, kill_container)
                return SandboxResult(
                    output="",
                    error=f"Execution timeout after {timeout}s",
                    success=False,
                    execution_time=time.time() - start_time
                )
            except Exception as e:
                # Other error
                def kill_container():
                    try:
                        container.kill()
                    except Exception:
                        pass
                
                await loop.run_in_executor(None, kill_container)
                return SandboxResult(
                    output="",
                    error=f"Execution error: {str(e)}",
                    success=False,
                    execution_time=time.time() - start_time
                )
        except Exception as e:
            logger.error(f"Docker sandbox execution failed: {e}")
            return SandboxResult(
                output="",
                error=str(e),
                success=False
            )


class RestrictedSandbox(Sandbox):
    """RestrictedPython-based lightweight sandbox."""
    
    def __init__(self, allowed_modules: Optional[list] = None):
        """
        Initialize RestrictedPython sandbox.
        
        Args:
            allowed_modules: List of module names to allow (e.g., ["math", "sympy"])
        """
        self.allowed_modules = allowed_modules or ["math"]
        self._initialize_restricted()
    
    def _initialize_restricted(self):
        """Initialize RestrictedPython."""
        try:
            from RestrictedPython import compile_restricted, safe_globals
            self._compile_restricted = compile_restricted
            self._safe_globals = safe_globals
            logger.info("RestrictedPython sandbox initialized")
        except ImportError:
            logger.warning(
                "RestrictedPython not installed. Install with: pip install RestrictedPython"
            )
            raise
    
    async def execute(
        self,
        code: str,
        timeout: int = 5,
        language: str = "python",
        **kwargs
    ) -> SandboxResult:
        """Execute code using RestrictedPython."""
        import time
        import io
        from contextlib import redirect_stdout, redirect_stderr
        start_time = time.time()
        
        try:
            # Build safe globals with allowed modules
            safe_globals_dict = self._safe_globals.copy()
            
            for module_name in self.allowed_modules:
                try:
                    module = __import__(module_name)
                    safe_globals_dict[module_name] = module
                except ImportError:
                    logger.debug(f"Module {module_name} not available")
            
            # Compile with restrictions - try 'eval' first, fallback to 'exec'
            byte_code = self._compile_restricted(code, '<inline>', 'eval')
            
            if byte_code.errors:
                # Try 'exec' mode instead
                byte_code = self._compile_restricted(code, '<inline>', 'exec')
                if byte_code.errors:
                    return SandboxResult(
                        output="",
                        error=f"Compilation errors: {byte_code.errors}",
                        success=False
                    )
                is_exec_mode = True
            else:
                is_exec_mode = False
            
            # Capture stdout/stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            def execute_code():
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    if is_exec_mode:
                        exec(byte_code.code, safe_globals_dict, {})
                        result = None
                    else:
                        result = eval(byte_code.code, safe_globals_dict, {})
                return result
            
            # Execute with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(execute_code),
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            # Combine stdout and result
            output = stdout_capture.getvalue().strip()
            if result is not None:
                output = f"{output}\n{result}".strip() if output else str(result)
            
            stderr_output = stderr_capture.getvalue().strip()
            if stderr_output:
                return SandboxResult(
                    output=output,
                    error=stderr_output,
                    success=False,
                    execution_time=execution_time
                )
            
            return SandboxResult(
                output=output or "",
                success=True,
                execution_time=execution_time
            )
        except asyncio.TimeoutError:
            return SandboxResult(
                output="",
                error=f"Execution timeout after {timeout}s",
                success=False,
                execution_time=timeout
            )
        except Exception as e:
            logger.warning(f"RestrictedPython execution failed: {e}")
            return SandboxResult(
                output="",
                error=str(e),
                success=False,
                execution_time=time.time() - start_time
            )


class SubprocessSandbox(Sandbox):
    """Basic subprocess-based sandbox (fallback, less secure)."""
    
    def __init__(self):
        """Initialize subprocess sandbox."""
        logger.info("Subprocess sandbox initialized (basic isolation)")
    
    async def execute(
        self,
        code: str,
        timeout: int = 5,
        language: str = "python",
        **kwargs
    ) -> SandboxResult:
        """Execute code in subprocess with resource limits."""
        import time
        start_time = time.time()
        
        try:
            # Set resource limits (Linux/Unix only)
            if HAS_RESOURCE:
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
                    resource.setrlimit(resource.RLIMIT_AS, (100 * 1024 * 1024, 100 * 1024 * 1024))  # 100MB
                except (OSError, ValueError):
                    # Limits not supported on this system
                    pass
            # On Windows, resource limits are not available, but timeout is still enforced via asyncio.wait_for
            
            # Run in subprocess
            process = await asyncio.create_subprocess_exec(
                "python", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={"PYTHONPATH": ""}  # Minimal environment
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                execution_time = time.time() - start_time
                
                if process.returncode == 0:
                    return SandboxResult(
                        output=stdout.decode('utf-8').strip(),
                        success=True,
                        execution_time=execution_time
                    )
                else:
                    return SandboxResult(
                        output="",
                        error=stderr.decode('utf-8').strip() or f"Process exited with code {process.returncode}",
                        success=False,
                        execution_time=execution_time
                    )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return SandboxResult(
                    output="",
                    error=f"Execution timeout after {timeout}s",
                    success=False,
                    execution_time=timeout
                )
        except Exception as e:
            logger.error(f"Subprocess sandbox execution failed: {e}")
            return SandboxResult(
                output="",
                error=str(e),
                success=False
            )


def create_sandbox(sandbox_type: str = "subprocess", **config) -> Sandbox:
    """
    Create sandbox instance.
    
    Args:
        sandbox_type: Type of sandbox ("modal", "docker", "restricted", "subprocess")
        **config: Sandbox-specific configuration
    
    Returns:
        Sandbox instance
    
    Examples:
        sandbox = create_sandbox("docker", image="python:3.11-slim")
        sandbox = create_sandbox("modal")
        sandbox = create_sandbox("restricted", allowed_modules=["math", "sympy"])
    """
    if sandbox_type == "modal":
        return ModalSandbox(**config)
    elif sandbox_type == "docker":
        return DockerSandbox(**config)
    elif sandbox_type == "restricted":
        return RestrictedSandbox(**config)
    elif sandbox_type == "subprocess":
        return SubprocessSandbox(**config)
    else:
        raise ValueError(
            f"Unknown sandbox type: {sandbox_type}. "
            f"Supported types: 'modal', 'docker', 'restricted', 'subprocess'"
        )


