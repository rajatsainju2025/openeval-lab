"""
Jinja2 prompt templating system for OpenEval.

This module provides advanced prompt templating using Jinja2,
inspired by lm-evaluation-harness and OpenAI evals.
"""

import re
from typing import Any, Dict, Optional, Union

try:
    import jinja2
    from jinja2 import nodes
    JINJA2_AVAILABLE = True
except ImportError:
    jinja2 = None
    nodes = None
    JINJA2_AVAILABLE = False


class PromptTemplate:
    """Advanced prompt template using Jinja2."""
    
    def __init__(self, template: str, template_type: str = "auto"):
        """
        Initialize prompt template.
        
        Args:
            template: Template string (Jinja2 or basic format)
            template_type: 'jinja2', 'format', or 'auto'
        """
        self.template = template
        self.template_type = template_type
        
        if template_type == "auto":
            self.template_type = self._detect_template_type(template)
        
        if self.template_type == "jinja2" and not JINJA2_AVAILABLE:
            raise ImportError("jinja2 is required for Jinja2 templates. Install with: pip install jinja2")
        
        if self.template_type == "jinja2" and jinja2 is not None:
            self.jinja_env = jinja2.Environment(
                loader=jinja2.BaseLoader(),
                undefined=jinja2.ChainableUndefined,  # Allow undefined variables
                trim_blocks=True,
                lstrip_blocks=True
            )
            # Add custom filters
            self.jinja_env.filters['strip'] = str.strip
            self.jinja_env.filters['upper'] = str.upper
            self.jinja_env.filters['lower'] = str.lower
            self.jinja_env.filters['alpha'] = alpha_filter
            self.jinja_template = self.jinja_env.from_string(template)
    
    def _detect_template_type(self, template: str) -> str:
        """Auto-detect template type based on syntax."""
        # Check for Jinja2 patterns
        jinja_patterns = [
            r'\{\{.*?\}\}',  # {{ variable }}
            r'\{%.*?%\}',    # {% for %}, {% if %}, etc.
            r'\{#.*?#\}',    # {# comment #}
        ]
        
        for pattern in jinja_patterns:
            if re.search(pattern, template):
                return "jinja2"
        
        # Check for format patterns
        if '{' in template and '}' in template:
            return "format"
        
        return "format"
    
    def render(self, **kwargs) -> str:
        """Render template with provided variables."""
        if self.template_type == "jinja2":
            return self.jinja_template.render(**kwargs)
        else:
            return self.template.format(**kwargs)
    
    def get_variables(self) -> list[str]:
        """Get list of template variables."""
        if self.template_type == "jinja2" and jinja2 is not None:
            # Parse Jinja2 template to find variables
            ast = self.jinja_env.parse(self.template)
            variables = []
            
            def find_variables(node):
                if nodes is not None and isinstance(node, nodes.Name):
                    variables.append(node.name)
                for child in node.iter_child_nodes():
                    find_variables(child)
            
            find_variables(ast)
            return list(set(variables))
        else:
            # Parse format string to find variables
            import string
            formatter = string.Formatter()
            return [field_name for _, field_name, _, _ in formatter.parse(self.template) if field_name]


def build_prompt(
    template: Union[str, PromptTemplate],
    variables: Dict[str, Any],
    template_type: str = "auto"
) -> str:
    """
    Build prompt from template and variables.
    
    Args:
        template: Template string or PromptTemplate instance
        variables: Variables to substitute
        template_type: Template type if template is string
    
    Returns:
        Rendered prompt string
    """
    if isinstance(template, str):
        template = PromptTemplate(template, template_type)
    
    return template.render(**variables)


# Example templates for common evaluation patterns
COMMON_TEMPLATES = {
    "multiple_choice": """{{question.strip()}}
{% for choice in choices %}
{{loop.index0 | alpha}}. {{choice}}
{% endfor %}
Answer:""",
    
    "qa": """Question: {{question}}
Answer:""",
    
    "classification": """{{text}}

Classify this text as: {{", ".join(labels)}}
Classification:""",
    
    "few_shot": """{% if examples %}
{% for example in examples %}
{{example.input}}
{{example.output}}

{% endfor %}
{% endif %}
{{input}}""",
    
    "chain_of_thought": """{{question}}

Let's think step by step:""",
    
    "instruct": """{% if system_prompt %}{{system_prompt}}

{% endif %}{{instruction}}{% if input %}

Input: {{input}}{% endif %}

Response:""",
}


def alpha_filter(index: int) -> str:
    """Convert index to alphabetic label (A, B, C, ...)."""
    return chr(ord('A') + index)


# Register the alpha filter if Jinja2 is available
if JINJA2_AVAILABLE and jinja2 is not None:
    # Custom filter is already added in PromptTemplate.__init__
    pass
