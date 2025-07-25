
"""
LLM Client for AI-powered Research Paper Analysis
Supports multiple providers: OpenAI, Anthropic, and local models
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"

@dataclass
class LLMResponse:
    """Standardized response from LLM providers"""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    response_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class LLMClientError(Exception):
    """Custom exception for LLM client errors"""
    pass

class LLMClient:
    """Unified client for multiple LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients based on available providers
        self.clients = {}
        self._initialize_clients()
        
        # Set default provider
        self.default_provider = self._determine_default_provider()
        
    def _initialize_clients(self):
        """Initialize available LLM clients"""
        
        # OpenAI client
        if OPENAI_AVAILABLE and self.config.get('openai', {}).get('api_key'):
            try:
                self.clients[LLMProvider.OPENAI] = openai.OpenAI(
                    api_key=self.config['openai']['api_key']
                )
                self.logger.info("OpenAI client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        # Anthropic client
        if ANTHROPIC_AVAILABLE and self.config.get('anthropic', {}).get('api_key'):
            try:
                self.clients[LLMProvider.ANTHROPIC] = anthropic.Anthropic(
                    api_key=self.config['anthropic']['api_key']
                )
                self.logger.info("Anthropic client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Anthropic client: {e}")
        
        # Local model client (if configured)
        if self.config.get('local', {}).get('endpoint'):
            if REQUESTS_AVAILABLE:
                self.clients[LLMProvider.LOCAL] = {
                    'endpoint': self.config['local']['endpoint'],
                    'headers': self.config['local'].get('headers', {})
                }
                self.logger.info("Local model client configured")
            else:
                self.logger.warning("Requests library not available for local model client")
    
    def _determine_default_provider(self) -> LLMProvider:
        """Determine the default provider based on availability"""
        if LLMProvider.OPENAI in self.clients:
            return LLMProvider.OPENAI
        elif LLMProvider.ANTHROPIC in self.clients:
            return LLMProvider.ANTHROPIC
        elif LLMProvider.LOCAL in self.clients:
            return LLMProvider.LOCAL
        else:
            raise LLMClientError("No LLM providers available. Please configure at least one provider.")
    
    def generate(
        self,
        prompt: str,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using specified or default provider"""
        
        provider = provider or self.default_provider
        start_time = time.time()
        
        try:
            if provider == LLMProvider.OPENAI:
                return self._generate_openai(prompt, model, max_tokens, temperature, system_prompt, **kwargs)
            elif provider == LLMProvider.ANTHROPIC:
                return self._generate_anthropic(prompt, model, max_tokens, temperature, system_prompt, **kwargs)
            elif provider == LLMProvider.LOCAL:
                return self._generate_local(prompt, model, max_tokens, temperature, system_prompt, **kwargs)
            else:
                raise LLMClientError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"LLM generation failed after {response_time:.2f}s: {e}")
            raise LLMClientError(f"Failed to generate response: {e}")
    
    def _generate_openai(self, prompt: str, model: Optional[str], max_tokens: Optional[int], 
                        temperature: float, system_prompt: Optional[str], **kwargs) -> LLMResponse:
        """Generate response using OpenAI"""
        
        if LLMProvider.OPENAI not in self.clients:
            raise LLMClientError("OpenAI client not available")
        
        client = self.clients[LLMProvider.OPENAI]
        model = model or self.config.get('openai', {}).get('default_model', 'gpt-3.5-turbo')
        max_tokens = max_tokens or self.config.get('openai', {}).get('max_tokens', 8000)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        response_time = time.time() - start_time
        
        return LLMResponse(
            content=response.choices[0].message.content,
            provider="openai",
            model=model,
            tokens_used=response.usage.total_tokens if response.usage else None,
            response_time=response_time,
            metadata={"finish_reason": response.choices[0].finish_reason}
        )
    
    def _generate_anthropic(self, prompt: str, model: Optional[str], max_tokens: Optional[int],
                           temperature: float, system_prompt: Optional[str], **kwargs) -> LLMResponse:
        """Generate response using Anthropic"""
        
        if LLMProvider.ANTHROPIC not in self.clients:
            raise LLMClientError("Anthropic client not available")
        
        client = self.clients[LLMProvider.ANTHROPIC]
        model = model or self.config.get('anthropic', {}).get('default_model', 'claude-3-sonnet-20240229')
        max_tokens = max_tokens or self.config.get('anthropic', {}).get('max_tokens', 8000)
        
        start_time = time.time()
        
        message_params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            message_params["system"] = system_prompt
        
        response = client.messages.create(**message_params, **kwargs)
        
        response_time = time.time() - start_time
        
        return LLMResponse(
            content=response.content[0].text,
            provider="anthropic",
            model=model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens if response.usage else None,
            response_time=response_time,
            metadata={"stop_reason": response.stop_reason}
        )
    
    def _generate_local(self, prompt: str, model: Optional[str], max_tokens: Optional[int],
                       temperature: float, system_prompt: Optional[str], **kwargs) -> LLMResponse:
        """Generate response using local model"""
        
        if LLMProvider.LOCAL not in self.clients:
            raise LLMClientError("Local model client not available")
        
        client_config = self.clients[LLMProvider.LOCAL]
        model = model or self.config.get('local', {}).get('default_model', 'local-model')
        max_tokens = max_tokens or self.config.get('local', {}).get('max_tokens', 8000)
        
        # Prepare request payload (adjust based on your local model API)
        payload = {
            "model": model,
            "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        start_time = time.time()
        
        response = requests.post(
            client_config['endpoint'],
            json=payload,
            headers=client_config['headers'],
            timeout=self.config.get('local', {}).get('timeout', 60)
        )
        
        response.raise_for_status()
        response_data = response.json()
        
        response_time = time.time() - start_time
        
        # Extract content (adjust based on your local model response format)
        content = response_data.get('choices', [{}])[0].get('text', '') or response_data.get('response', '')
        
        return LLMResponse(
            content=content,
            provider="local",
            model=model,
            response_time=response_time,
            metadata=response_data
        )
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [provider.value for provider in self.clients.keys()]
    
    def test_connection(self, provider: Optional[LLMProvider] = None) -> bool:
        """Test connection to specified or default provider"""
        provider = provider or self.default_provider
        
        try:
            response = self.generate(
                "Hello, this is a test message. Please respond with 'Test successful.'",
                provider=provider,
                max_tokens=50,
                temperature=0.1
            )
            return "test successful" in response.content.lower()
        except Exception as e:
            self.logger.error(f"Connection test failed for {provider.value}: {e}")
            return False

def create_llm_client(config_path: Optional[str] = None) -> LLMClient:
    """Factory function to create LLM client with configuration"""
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                config = json.load(f)
            else:  # Assume YAML
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    raise LLMClientError("PyYAML required for YAML config files")
    else:
        # Default configuration from environment variables
        config = {
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'default_model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                'max_tokens': int(os.getenv('OPENAI_MAX_TOKENS', '8000'))
            },
            'anthropic': {
                'api_key': os.getenv('ANTHROPIC_API_KEY'),
                'default_model': os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
                'max_tokens': int(os.getenv('ANTHROPIC_MAX_TOKENS', '8000'))
            },
            'local': {
                'endpoint': os.getenv('LOCAL_LLM_ENDPOINT'),
                'default_model': os.getenv('LOCAL_LLM_MODEL', 'local-model'),
                'max_tokens': int(os.getenv('LOCAL_LLM_MAX_TOKENS', '8000')),
                'timeout': int(os.getenv('LOCAL_LLM_TIMEOUT', '60')),
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': f"Bearer {os.getenv('LOCAL_LLM_API_KEY', '')}"
                } if os.getenv('LOCAL_LLM_API_KEY') else {'Content-Type': 'application/json'}
            }
        }
    
    return LLMClient(config)
